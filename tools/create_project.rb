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
#SBATCH --mem-per-cpu=1000
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

options = {}
option_parser = OptionParser.new do |opts|
  opts.banner = "Usage: create_project.rb [options]"

  opts.on("-n", "--name PROJECT_NAME", "Project name") do |name|
    options[:project_name] = name
  end
  opts.on("--mode MODE", "Mode (e+n, p+n, e+p+n, or e+n?)") do |v|
    if ["e+n", "p+n", "e+p+n", "e+n?"].include?(v)
      options[:mode] = v
    else
      puts "Invalid mode. Allowed values are 'e+n', 'p+n', 'e+p+n', or 'e+n?'."
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

# Read the .gcard file
gcard_path = options[:gcard]
unless File.exist?(gcard_path)
  puts "Gcard file does not exist: #{gcard_path}"
  exit
end

# Read the grid .yaml file
grid_yaml_path = options[:grid]
unless File.exist?(grid_yaml_path)
  puts "Grid YAML file does not exist: #{grid_yaml_path}"
  exit
end

# Read the recon .yaml file
recon_yaml_path = options[:recon]
unless File.exist?(recon_yaml_path)
  puts "Recon YAML file does not exist: #{recon_yaml_path}"
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
  file_slurm = file_lund.gsub('/lund/','/slurm/').gsub('.lund','.slurm')
    
  puts "Generated slurm file: #{file_slurm}"
  slurm_command = "bash ./tools/pipeline.sh #{gcard_path} #{file_lund} #{file_gemc} #{file_cooked} #{file_dst} #{file_train} #{recon_yaml_path}"
  create_slurm_file(slurm_command, file_slurm, options[:project_name])
  system("sbatch #{file_slurm}")
end
