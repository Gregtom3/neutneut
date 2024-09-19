#!/usr/bin/env ruby
require 'fileutils'
require 'date'
require 'optparse'
require 'yaml'

abort("This script must be run from the 'neutneut' directory") unless File.basename(Dir.pwd) == "neutneut"

def create_slurm_file(command, slurm_filename, job_name)
  slurm_template = <<-SLURM
#!/bin/bash
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=#{job_name}
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=#{slurm_filename}.out
#SBATCH --error=#{slurm_filename}.err

#{command}
  SLURM

  File.open(slurm_filename, "w") { |file| file.write(slurm_template) }
  slurm_filename
end

options = {
  gcard: 'gcards/rga_fall2018.gcard',
  grid: 'grids/simple_grid_neutron.yaml',
  recon: 'recon/rga_fall2018.yaml'
}

option_parser = OptionParser.new do |opts|
  opts.banner = "Usage: create_project.rb [options]"

  opts.on("-n", "--name PROJECT_NAME", "Project name") do |name|
    options[:project_name] = name
  end
  opts.on("--mode MODE", "Mode (e+n, p+n, e+p+n, e+n?, or dis)") do |v|
    if ["e+n", "p+n", "e+p+n", "e+n?", "dis"].include?(v)
      options[:mode] = v
    else
      puts "Invalid mode. Allowed values are 'e+n', 'p+n', 'e+p+n', 'e+n?', or 'dis'."
      exit
    end
  end
  opts.on("--gcard GCARD", "Name of the .gcard file") do |v|
    options[:gcard] = v
  end
  opts.on("--grid GRID", "YAML configuration file for grid setup") do |grid|
    options[:grid] = grid
  end
  opts.on("--recon RECON", "YAML configuration file for the recon-util reconstruction") do |recon|
    options[:recon] = recon
  end
end

option_parser.parse!

# If no gcard is provided, list available gcards
if options[:gcard].nil?
  gcard_files = Dir.glob("./gcards/*.gcard")
  puts "Available gcards (use --gcard):"
  gcard_files.each { |file| puts "  ./gcards/#{File.basename(file)}" }
end

# If no grid is provided, list available grids
if options[:grid].nil?
  grid_files = Dir.glob("./grids/*.yaml")
  puts "Available simulation grids (use --grid):"
  grid_files.each { |file| puts "  ./grids/#{File.basename(file)}" }
end

# If no recon is provided, list available recons
if options[:recon].nil?
  yaml_files = Dir.glob("./recon/*.yaml")
  puts "Available recon yamls (use --recon):"
  yaml_files.each { |file| puts "  ./recon/#{File.basename(file)}" }
end

if options[:project_name].nil? || options[:grid].nil? || options[:gcard].nil? || options[:recon].nil?
  puts option_parser
  exit
end

# Get current date and time
current_time = Time.now
timestamp = current_time.strftime("%m.%d.%Y.%H.%M")

# Create the project directory path
project_name = options[:project_name]
project_dir = "./projects/#{project_name}.#{timestamp}"

# Function to create project directories
def create_project_dirs(base_dir)
  subdirs = ['gemc', 'cooked', 'dst', 'slurm', 'lund', 'training']
  subdirs.each do |subdir|
    Dir.mkdir(File.join(base_dir, subdir))
  end
end

# Check if directory already exists
if Dir.exist?(project_dir)
  puts "Directory #{project_dir} already exists. Do you want to overwrite it? (y/n)"
  answer = gets.chomp.downcase
  if answer == 'y'
    FileUtils.rm_rf(project_dir) # Remove existing directory
    Dir.mkdir(project_dir) # Create a new directory
    create_project_dirs(project_dir) # Create subdirectories
    puts "Directory #{project_dir} and subdirectories created successfully."
  else
    puts "Directory not overwritten. Exiting."
    exit
  end
else
  Dir.mkdir(project_dir) # Create a new directory
  create_project_dirs(project_dir) # Create subdirectories
  puts "Directory #{project_dir} and subdirectories created successfully."
end


# Store Slurm job IDs
slurm_job_ids = []

# If the mode is "dis", prompt for the number of events and batches
if options[:mode] == "dis"
  print "Enter the number of events: "
  nevents = gets.chomp.to_i

  print "Enter the number of batches: "
  num_batches = gets.chomp.to_i

  nevents_per_batch = (nevents / num_batches.to_f).ceil

  # Change to the project directory
  Dir.chdir(project_dir) do
    system("mkdir eventfiles")
    system("clasdis --trig #{nevents} --nmax #{nevents_per_batch}")
    # Move all .dat files from eventfiles to lund and delete eventfiles directory
    dat_files = Dir.glob("eventfiles/*.dat")
    dat_files.each do |dat_file|
      FileUtils.mv(dat_file, "lund/")
    end
    FileUtils.rm_rf("eventfiles")
  end

  # Create slurm files for the .dat files in the lund directory
  Dir.glob("#{project_dir}/lund/*.dat").each do |file_lund|
    file_gemc = file_lund.gsub('/lund/', '/gemc/').gsub('.dat','.hipo')
    file_cooked = file_lund.gsub('/lund/', '/cooked/').gsub('.dat','.hipo')
    file_dst   = file_lund.gsub('/lund/', '/dst/').gsub('.dat','.hipo')
    file_train   = file_lund.gsub('/lund/', '/training/').gsub('.dat','.csv')
    file_h5    = file_lund.gsub('/lund/', '/training/').gsub('.dat','.h5')
    file_slurm = file_lund.gsub('/lund/','/slurm/').gsub('.dat','.slurm')
    
    slurm_command = "bash ./tools/pipeline.sh #{options[:gcard]} #{file_lund} #{file_gemc} #{file_cooked} #{file_dst} #{file_train} #{file_h5} #{options[:recon]}"
    create_slurm_file(slurm_command, file_slurm, options[:project_name])
    job_id = `sbatch #{file_slurm}`.strip.split.last
    slurm_job_ids << job_id
  end
else
  # Read the YAML configuration file
  config = YAML.load_file(options[:grid])

  particle = config['particle']
  nevents = config['Nevents'].to_i
  p_min = config['P']['min']
  p_max = config['P']['max']
  p_step = config['P']['step']
  theta_min = config['Theta']['min']
  theta_max = config['Theta']['max']
  theta_step = config['Theta']['step']
  phi_min = config['Phi']['min']
  phi_max = config['Phi']['max']
  phi_step = config['Phi']['step']
  spread_p = config['SPREAD_P']

  # Generate values and call generate_gcard_gun.rb
  output_dir = File.join(project_dir, 'lund')
  FileUtils.mkdir_p(output_dir)

  (p_min..p_max).step(p_step).each do |p|
    (theta_min..theta_max).step(theta_step).each do |theta|
      (phi_min..phi_max).step(phi_step).each do |phi|
        system("ruby ./tools/generate_lund_events.rb --output #{output_dir} --mode #{options[:mode]} --nevents #{nevents} --beam_p \"#{particle}, #{p}*GeV, #{theta}*deg, #{phi}*deg\" --spread_p \"#{spread_p}\"")
      end
    end
  end

  # Print all the files in the output_dir
  Dir.glob(File.join(output_dir, '**', '*')).each do |file|
    file_lund = file
    file_gemc = file_lund.gsub('/lund/', '/gemc/').gsub('.lund','.hipo')
    file_cooked = file_lund.gsub('/lund/', '/cooked/').gsub('.lund','.hipo')
    file_dst   = file_lund.gsub('/lund/', '/dst/').gsub('.lund','.hipo')
    file_train   = file_lund.gsub('/lund/', '/training/').gsub('.lund','.csv')
    file_h5    = file_lund.gsub('/lund/', '/training/').gsub('.lund','.h5')
    file_slurm = file_lund.gsub('/lund/','/slurm/').gsub('.lund','.slurm')
      
    puts "Generated slurm file: #{file_slurm}"
    slurm_command = "bash ./tools/pipeline.sh #{options[:gcard]} #{file_lund} #{file_gemc} #{file_cooked} #{file_dst} #{file_train} #{file_h5} #{options[:recon]}"
    create_slurm_file(slurm_command, file_slurm, options[:project_name])
    job_id = `sbatch #{file_slurm}`.strip.split.last
    slurm_job_ids << job_id
  end
end

# NO LONGER NEEDED 

# # Create a Slurm file for combining .h5 files
# h5_dir = File.join(project_dir, 'training')
# combine_command = "python3 tools/combine_h5.py #{h5_dir}"
# combine_slurm_filename = File.join(project_dir, "slurm/combine_h5.slurm")

# combine_slurm_content = <<-SLURM
# #!/bin/bash
# #SBATCH --account=clas12
# #SBATCH --partition=production
# #SBATCH --mem-per-cpu=2000
# #SBATCH --job-name=combine_h5
# #SBATCH --cpus-per-task=1
# #SBATCH --time=2:00:00
# #SBATCH --output=#{combine_slurm_filename}.out
# #SBATCH --error=#{combine_slurm_filename}.err
# #SBATCH --dependency=afterok:#{slurm_job_ids.join(':')}

# #{combine_command}
# SLURM

# File.open(combine_slurm_filename, "w") { |file| file.write(combine_slurm_content) }

# # Submit the combine job after all other jobs
# system("sbatch #{combine_slurm_filename}")
