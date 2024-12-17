#!/usr/bin/env ruby
require 'fileutils'
require 'yaml'

# Function to get the 5 most recently created directories in a given path
def recent_directories(path)
  Dir.glob("#{path}/*").select { |f| File.directory?(f) }.sort_by { |f| File.mtime(f) }.reverse.first(5)
end

# Function to find the latest checkpoint in a given directory
def find_latest_checkpoint(checkpoint_dir)
  checkpoints = Dir.glob("#{checkpoint_dir}/*.keras*").sort_by { |f| File.mtime(f) }
  return checkpoints.last if checkpoints.any?

  nil
end

# Helper function to convert input to the correct type based on the original value
def convert_to_type(input, original_value)
  return input if input.strip.upcase == "N"
  
  case original_value
  when Integer
    input.to_i
  when Float
    input.to_f
  when TrueClass, FalseClass
    input.strip.downcase == "true"
  else
    input
  end
end

# Function to create a new episode directory
def create_episode_directory(base_path)
  FileUtils.mkdir_p(base_path) unless Dir.exist?(base_path)
  episode_number = 0
  loop do
    episode_dir = File.join(base_path, format("episode_%04d", episode_number))
    unless Dir.exist?(episode_dir)
      FileUtils.mkdir_p(episode_dir)
      return episode_dir
    end
    episode_number += 1
  end
end

# Function to prompt the user for a YAML configuration file and allow modifications
def prompt_for_config(config_dir, training_dir, episode_dir)
  configs = Dir.glob("#{config_dir}/*.yaml").map { |f| File.basename(f) }
  puts "Available configurations:"
  configs.each_with_index { |config, i| puts "#{i + 1}. #{config}" }

  config_index = nil
  loop do
    print "Select a configuration by number: "
    input = STDIN.gets.chomp
    config_index = input.to_i - 1
    break if config_index.between?(0, configs.length - 1)

    puts "Invalid selection. Please select a number between 1 and #{configs.length}."
  end

  selected_config = configs[config_index]
  config_path = File.join(config_dir, selected_config)
  config_data = YAML.load_file(config_path)
  config_data['project_directory'] = training_dir
  config_data['output_dir'] = episode_dir
  puts "Selected configuration:"
  puts YAML.dump(config_data)

  loop do
    puts "Current configuration:"
    puts YAML.dump(config_data)

    print "Do you want to change any parameter? (Y/N): "
    response = STDIN.gets.chomp.strip.upcase
    break if response == "N"

    puts "Available parameters to change:"
    editable_keys = config_data.keys.reject { |key| %w[project_directory output_dir].include?(key) }
    editable_keys.each_with_index do |key, index|
      puts "#{index + 1}. #{key} (current value: #{config_data[key]})"
    end

    print "Enter the parameter number to change: "
    param_index = STDIN.gets.chomp.to_i - 1
    param_key = editable_keys[param_index]

    if config_data[param_key].is_a?(Hash)
      config_data[param_key].each do |sub_key, sub_value|
        print "Change #{sub_key} (default #{sub_value})? (N to skip): "
        input = STDIN.gets.chomp
        config_data[param_key][sub_key] = convert_to_type(input, sub_value) unless input.strip.upcase == "N"
      end
    else
      print "Change #{param_key} (default #{config_data[param_key]})? (N to skip): "
      input = STDIN.gets.chomp
      config_data[param_key] = convert_to_type(input, config_data[param_key]) unless input.strip.upcase == "N"
    end
  end

  config_data
end

# Function to prompt the user for grid search parameters
def prompt_for_grid_search(config_data, episode_dir)
  grid_search_params = {}
  grid_search_enabled = false
  
  loop do
    puts "Current configuration:"
    puts YAML.dump(config_data)
    print "Do you want to grid search any parameter? (Y/N): "
    response = STDIN.gets.chomp.strip.upcase
    break if response == "N"
    
    print "Enter the parameter to grid (e.g., batch_size, N_grav_layers): "
    param = STDIN.gets.chomp

    if config_data.key?(param)
      values = []
      loop do
        print "Enter a value for #{param} (or 'done' to finish): "
        value = STDIN.gets.chomp
        break if value.strip.downcase == "done"
        grid_search_enabled = true
        values << convert_to_type(value, config_data[param])
      end

      if values.any?
        grid_search_params[param] = values
      end
    else
      puts "Parameter #{param} not found in configuration."
    end
  end

  # Generate all combinations of grid search parameters if any were specified
  if grid_search_enabled
    generate_grid_search_configs(config_data, grid_search_params, episode_dir)
  end

  return grid_search_enabled
end

# Helper function to generate all combinations of grid search configs
def generate_grid_search_configs(config_data, grid_search_params, episode_dir)
  keys = grid_search_params.keys
  values_combinations = keys.map { |key| grid_search_params[key] }
  
  # Generate all combinations using product, then map the results correctly
  combinations = values_combinations.shift.product(*values_combinations).map(&:flatten)

  combinations.each_with_index do |values, idx|
    config_copy = config_data.clone
    keys.each_with_index do |key, key_idx|
      config_copy[key] = values[key_idx]
    end

    config_subdir = File.join(episode_dir, format("config_%04d", idx))
    FileUtils.mkdir_p(config_subdir)
    config_copy["output_dir"] = config_subdir
    config_path = File.join(config_subdir, "config.yaml")
    File.write(config_path, YAML.dump(config_copy))

    # Create and submit SLURM script
    create_and_submit_slurm_script(config_subdir, idx)
  end
end
# Function to create and submit SLURM script with optional checkpoint path
def create_and_submit_slurm_script(config_subdir, config_idx, checkpoint_path = nil)
  checkpoint_arg = checkpoint_path ? "--checkpoint #{checkpoint_path}" : ""

  slurm_script = <<~SLURM
    #!/bin/bash
    #SBATCH --account=clas12
    #SBATCH --partition=gpu
    #SBATCH --mem-per-cpu=4000
    #SBATCH --job-name=neutneut_episode_#{File.basename(File.dirname(config_subdir))}_config_#{format('%04d', config_idx)}
    #SBATCH --cpus-per-task=4
    #SBATCH --time=24:00:00
    #SBATCH --gres=gpu:TitanRTX:1
    #SBATCH --output=#{config_subdir}/slurm.out
    #SBATCH --error=#{config_subdir}/slurm.err
    
    srun --pty bash
    source ~/.bashrc
    module purge
    module load python/3.12.4
    source /work/clas12/users/gmat/venv/tensorflow_env/bin/activate
    which python
    whereis python
    python --version
    pip install hipopy
    python ./tools/train_model.py #{config_subdir}/config.yaml #{checkpoint_arg}
  SLURM

  slurm_path = File.join(config_subdir, "slurm.slurm")
  File.write(slurm_path, slurm_script)

  # Submit the SLURM job
  system("sbatch #{slurm_path}")
end

# Main program logic
def main
  # Parse command line arguments
  continue_from_checkpoint = nil
  if ARGV.length >= 2 && ARGV[0] == '--name'
    project_name = ARGV[1]
    
    # Check for optional --continue_from_checkpoint argument
    if ARGV.length == 4 && ARGV[2] == '--continue_from_checkpoint'
      continue_from_checkpoint = ARGV[3]
    end
  else
    puts "Usage: ruby train_project.rb --name <NAME> [--continue_from_checkpoint <CHECKPOINT_PATH>]"
    exit(1)
  end

  project_path = File.join('./projects', project_name)
  
  unless Dir.exist?(project_path)
    puts "Project directory not found: #{project_path}"
    puts "Here are the 5 most recently created project directories:"
    recent_directories('./projects').each { |dir| puts File.basename(dir) }
    exit(1)
  end

  # Determine the episode directory and configuration directory
  if continue_from_checkpoint
    # Extract the episode and config directory from the checkpoint path
    episode_dir = File.dirname(File.dirname(continue_from_checkpoint))
    config_subdir = File.dirname(File.dirname(continue_from_checkpoint))
    config_path = File.join(config_subdir, "config.yaml")
    # Load configuration directly from specified checkpoint's config.yaml
    if File.exist?(config_path)
      config_data = YAML.load_file(config_path)
      puts "Loaded configuration from #{config_path} for continued training."
    else
      puts "Configuration file not found: #{config_path}"
      exit(1)
    end
  else
    # Create a new episode directory if not continuing from checkpoint
    episode_dir = create_episode_directory(File.join(project_path, 'tensorflow'))
    training_dir = File.join(project_path, 'training')
    FileUtils.mkdir_p(training_dir)

    # Prompt user to select and modify configuration if not resuming from checkpoint
    config_data = prompt_for_config('./training_config', training_dir, episode_dir)
    config_subdir = File.join(episode_dir, 'config_0000')
    FileUtils.mkdir_p(config_subdir)
    config_path = File.join(config_subdir, "config.yaml")
    File.write(config_path, YAML.dump(config_data))

    puts "Initial configuration:"
    puts YAML.dump(config_data)
  end

  # Determine if grid search is enabled
  grid_search_enabled = continue_from_checkpoint ? false : prompt_for_grid_search(config_data, episode_dir)

  # Submit SLURM job
  unless grid_search_enabled
    # Only submit the single configuration in the checkpoint case
    create_and_submit_slurm_script(config_subdir, 0, continue_from_checkpoint)
  end

  puts "Training setup complete. Configuration saved in #{episode_dir}."
end




# Execute the main program
main
