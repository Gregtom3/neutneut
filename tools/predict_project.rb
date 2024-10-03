#!/usr/bin/env ruby
require 'fileutils'
require 'date'
require 'optparse'

# Parse command-line arguments
options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: predict_project.rb --name <NAME> --tB <VALUE> --tD <VALUE>"

  opts.on("--name NAME", "Project name to run predictions for") do |name|
    options[:project] = name
  end

  opts.on("--tB VALUE", Float, "Clustering threshold tB (must be between 0 and 1)") do |value|
    options[:tB] = value
  end

  opts.on("--tD VALUE", Float, "Clustering distance tD (must be a positive value)") do |value|
    options[:tD] = value
  end
end.parse!

# Ensure project name, tB, and tD are provided
unless options[:project]
  puts "Error: --project <NAME> is required."
  exit 1
end

unless options[:tB] && options[:tB] > 0 && options[:tB] <= 1
  puts "Error: --tB <VALUE> is required and must be between 0 and 1."
  exit 1
end

unless options[:tD] && options[:tD] > 0
  puts "Error: --tD <VALUE> is required and must be a positive value."
  exit 1
end

project_name = options[:project]
project_dir = "./projects/#{project_name}"
slurm_dir = "#{project_dir}/slurm"

# Ensure the project and slurm directories exist
FileUtils.mkdir_p(slurm_dir)
FileUtils.mkdir_p("#{project_dir}/predict")

# Function to create SLURM file
def create_slurm_file(slurm_filename, job_name, slurm_dir, slurm_commands)
  slurm_template = <<-SLURM
#!/bin/bash
#SBATCH --account=clas12
#SBATCH --partition=production
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=#{job_name}
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=#{slurm_dir}/#{slurm_filename}.out
#SBATCH --error=#{slurm_dir}/#{slurm_filename}.err

#{slurm_commands}
  SLURM

  File.open("#{slurm_dir}/#{slurm_filename}", "w") { |file| file.write(slurm_template) }
end

# Recursively search for 'trained_model.keras' files in the tensorflow directory
model_files = Dir.glob("#{project_dir}/tensorflow/**/*trained_model.keras")

# If no models found, exit with an error
if model_files.empty?
  puts "Error: No trained_model.keras files found in #{project_dir}/tensorflow/"
  exit 1
end

# Print the found model files and let the user select the model
puts "Select a model to use for prediction:"
model_files.each_with_index do |model_file, index|
  puts "#{index + 1}) #{model_file}"
end

print "Enter the number of the model to use: "
model_choice = gets.to_i

# Ensure a valid choice is made
if model_choice < 1 || model_choice > model_files.length
  puts "Error: Invalid model selection."
  exit 1
end

model_path = model_files[model_choice - 1]
puts "Using model: #{model_path}"

# Define Python program path
python_program = './tools/predict_model.py'

# Loop over all .h5 files in the training directory
Dir.glob("#{project_dir}/training/*.h5").each do |h5_file|
  # Extract the file's padded number from the h5 file name
  file_number = File.basename(h5_file, ".h5").split("_").first.split(".").last

  # Find the matching .hipo file with the same padded number
  hipo_file = "#{project_dir}/cooked/#{File.basename(h5_file, ".h5")}.hipo"

  unless File.exist?(hipo_file)
    puts "Error: Matching HIPO file for #{h5_file} not found (expected #{hipo_file})."
    next
  end

  # Define the paths for _OC.hipo, _OC1.hipo, and _OC2.hipo
  cooked_OC_orig_hipo = "#{project_dir}/cooked/#{File.basename(hipo_file, ".hipo")}_OC.hipo"
  cooked_OC_hipo = "#{project_dir}/predict/#{File.basename(hipo_file, ".hipo")}_OC.hipo"
  cooked_OC1_hipo = "#{project_dir}/predict/#{File.basename(hipo_file, ".hipo")}_OC1.hipo"
  cooked_OC2_hipo = "#{project_dir}/predict/#{File.basename(hipo_file, ".hipo")}_OC2.hipo"
  final_filtered_hipo = "#{project_dir}/predict/#{File.basename(hipo_file, ".hipo")}_ML.hipo"

  # SLURM commands to execute
  slurm_commands = <<-COMMANDS
# Step 0: Clean up intermediate files
echo "Cleaning up intermediate files..."
rm -f #{cooked_OC_hipo} #{cooked_OC1_hipo} #{cooked_OC2_hipo} #{final_filtered_hipo}

# Step 1: Run the Python program
echo "Running the Python program on #{h5_file}..."
python3 #{python_program} --input_h5 #{h5_file} --original_hipofile #{hipo_file} --clustering_variable unique_otid --tB #{options[:tB]} --tD #{options[:tD]} --model_path #{model_path}
mv #{cooked_OC_orig_hipo} #{cooked_OC_hipo}
if [ $? -ne 0 ]; then
  echo "Error: Python program execution failed for #{h5_file}."
  exit 1
fi

# Step 2: Apply hipo-utils filter to _OC.hipo
echo "Running hipo-utils filter on #{cooked_OC_hipo}..."
hipo-utils -filter -b 'COAT::config,DC::tdc,ECAL::adc,ECAL::calib,ECAL::calib_OC,ECAL::clusters,ECAL::clusters_OC,ECAL::hits,ECAL::hits+,ECAL::moments,ECAL::moments_OC,ECAL::peaks,ECAL::tdc,FTOF::adc,FTOF::clusters,FTOF::hbclusters,FTOF::hbhits,FTOF::hits,FTOF::matchedclusters,FTOF::rawhits,FTOF::tdc,HTCC::adc,HTCC::rec,HTCC::tdc,HitBasedTrkg::Clusters,HitBasedTrkg::HBClusters,HitBasedTrkg::HBCrosses,HitBasedTrkg::HBHitTrkId,HitBasedTrkg::HBHits,HitBasedTrkg::HBSegments,HitBasedTrkg::HBTracks,HitBasedTrkg::Hits,HitBasedTrkg::Trajectory,MC::Event,MC::GenMatch,MC::Lund,MC::Particle,MC::RecMatch,MC::True,RASTER::adc,RASTER::position,REC::CaloExtras,REC::Cherenkov,REC::CovMat,REC::Event,REC::ScintExtras,REC::Scintillator,REC::Track,REC::Traj,RECHB::CaloExtras,RECHB::Cherenkov,RECHB::Event,RECHB::Particle,RECHB::ScintExtras,RECHB::Scintillator,RECHB::Track,RECHB::Traj,RUN::config,RUN::rf,TimeBasedTrkg::TBClusters,TimeBasedTrkg::TBCovMat,TimeBasedTrkg::TBCrosses,TimeBasedTrkg::TBHits,TimeBasedTrkg::TBSegments,TimeBasedTrkg::TBTracks,TimeBasedTrkg::Trajectory,ai::tracks' #{cooked_OC_hipo} -o #{cooked_OC1_hipo}

if [ $? -ne 0 ]; then
  echo "Error: hipo-utils filter failed for #{cooked_OC_hipo}."
  exit 1
fi

# Step 3: Run recon-util-OC on _OC1.hipo
echo "Running recon-util-OC on #{cooked_OC1_hipo}..."
./tools/recon-util-OC -i #{cooked_OC1_hipo} -o #{cooked_OC2_hipo} -y ./recon/rga_fall2018_OC.yaml

if [ $? -ne 0 ]; then
  echo "Error: recon-util-OC failed for #{cooked_OC1_hipo}."
  exit 1
fi

# Step 4: Final filter to keep only specified banks
echo "Applying final hipo-utils filter to keep only specific banks..."
hipo-utils -filter -b 'RUN::*,MC::*,REC::Particle,REC::Calorimeter,REC::Track,REC::Traj,ECAL::*' #{cooked_OC2_hipo} -o #{final_filtered_hipo}

if [ $? -ne 0 ]; then
  echo "Error: Final filtering failed for #{cooked_OC2_hipo}."
  exit 1
fi

# Step 5: Clean up intermediate files
echo "Cleaning up intermediate files..."
rm -f #{cooked_OC_hipo} #{cooked_OC1_hipo} #{cooked_OC2_hipo}

echo "Processing complete for #{h5_file}. Final HIPO file saved at #{final_filtered_hipo}"
  COMMANDS

  # Create SLURM file
  slurm_filename = "predict_#{file_number}.slurm"
  create_slurm_file(slurm_filename, "predict_#{file_number}", slurm_dir, slurm_commands)

  # Submit the SLURM job
  puts "Submitting SLURM job for #{h5_file}..."
  system("sbatch #{slurm_dir}/#{slurm_filename}")

end

puts "\t ==> All SLURM jobs submitted successfully for project #{project_name}!"














