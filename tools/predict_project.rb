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

# Ensure the project directory exists
unless Dir.exist?(project_dir)
  puts "Error: Project directory #{project_dir} does not exist."
  exit 1
end

# Ensure the script is run from the "neutneut" directory
current_dir = File.basename(Dir.getwd)
if current_dir != 'neutneut'
  puts "Error: This script must be run from the 'neutneut' directory."
  exit 1
end

# Define paths for the project's training and predict directories
training_dir = "#{project_dir}/training"
predict_dir = "#{project_dir}/predict"

# Create the predict directory if it doesn't already exist
FileUtils.mkdir_p(predict_dir)

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

puts "#{training_dir}"
# Loop over all .h5 files in the training directory
Dir.glob("#{training_dir}/*.h5").each do |h5_file|
    
  # Extract the file's padded number from the h5 file name
  file_number = File.basename(h5_file, ".h5").split("_").first.split(".").last
  
  # Find the matching .hipo file with the same padded number
  hipo_file = "#{training_dir}/../cooked/#{File.basename(h5_file, ".h5")}.hipo"

  unless File.exist?(hipo_file)
    puts "Error: Matching HIPO file for #{h5_file} not found (expected #{hipo_file})."
    next
  end

  # Define the paths for _OC.hipo, _OC1.hipo, and _OC2.hipo
  cooked_OC_hipo = "#{predict_dir}/#{File.basename(hipo_file, ".hipo")}_OC.hipo"
  cooked_OC1_hipo = "#{predict_dir}/#{File.basename(hipo_file, ".hipo")}_OC1.hipo"
  cooked_OC2_hipo = "#{predict_dir}/#{File.basename(hipo_file, ".hipo")}_OC2.hipo"
  final_filtered_hipo = "#{predict_dir}/#{File.basename(hipo_file, ".hipo")}_ML.hipo"

  # Step 0: Clean up intermediate files
  # Avoids errors in recon-util
  puts "Cleaning up intermediate files..."
  File.delete(cooked_OC_hipo) if File.exist?(cooked_OC_hipo)
  File.delete(cooked_OC1_hipo) if File.exist?(cooked_OC1_hipo)
  File.delete(cooked_OC2_hipo) if File.exist?(cooked_OC2_hipo)
  File.delete(final_filtered_hipo) if File.exist?(final_filtered_hipo)
    
  # Step 1: Run the Python program with the .h5 and original hipo file
  puts "Running the Python program on #{h5_file} and #{hipo_file}..."
  python_command = "python3 #{python_program} --input_h5 #{h5_file} --original_hipofile #{hipo_file} --clustering_variable unique_otid --tB #{options[:tB]} --tD #{options[:tD]} --model_path #{model_path}"
  system(python_command)

  # Check if the Python program executed successfully
  if $?.exitstatus != 0
    puts "Error: Python program execution failed for #{h5_file}."
    next
  end

  # Rename the result of the Python program from original hipo file to _OC.hipo
  if File.exist?(hipo_file.gsub(".hipo", "_OC.hipo"))
    FileUtils.mv(hipo_file.gsub(".hipo", "_OC.hipo"), cooked_OC_hipo)
  else
    puts "Error: Expected output _OC.hipo not found for #{hipo_file}."
    next
  end

  # Step 2: Apply hipo-utils filter to _OC.hipo
  puts "Running hipo-utils filter on #{cooked_OC_hipo}..."
  hipo_filter_command = "hipo-utils -filter -b 'COAT::config,DC::tdc,ECAL::adc,ECAL::calib,ECAL::calib_OC,ECAL::clusters,ECAL::clusters_OC,ECAL::hits,ECAL::hits+,ECAL::moments,ECAL::moments_OC,ECAL::peaks,ECAL::tdc,FTOF::adc,FTOF::clusters,FTOF::hbclusters,FTOF::hbhits,FTOF::hits,FTOF::matchedclusters,FTOF::rawhits,FTOF::tdc,HTCC::adc,HTCC::rec,HTCC::tdc,HitBasedTrkg::Clusters,HitBasedTrkg::HBClusters,HitBasedTrkg::HBCrosses,HitBasedTrkg::HBHitTrkId,HitBasedTrkg::HBHits,HitBasedTrkg::HBSegments,HitBasedTrkg::HBTracks,HitBasedTrkg::Hits,HitBasedTrkg::Trajectory,MC::Event,MC::GenMatch,MC::Lund,MC::Particle,MC::RecMatch,MC::True,RASTER::adc,RASTER::position,REC::CaloExtras,REC::Cherenkov,REC::CovMat,REC::Event,REC::ScintExtras,REC::Scintillator,REC::Track,REC::Traj,RECHB::CaloExtras,RECHB::Cherenkov,RECHB::Event,RECHB::Particle,RECHB::ScintExtras,RECHB::Scintillator,RECHB::Track,RECHB::Traj,RUN::config,RUN::rf,TimeBasedTrkg::TBClusters,TimeBasedTrkg::TBCovMat,TimeBasedTrkg::TBCrosses,TimeBasedTrkg::TBHits,TimeBasedTrkg::TBSegments,TimeBasedTrkg::TBTracks,TimeBasedTrkg::Trajectory,ai::tracks' #{cooked_OC_hipo} -o #{cooked_OC1_hipo}"
  system(hipo_filter_command)

  # Check if the hipo-utils command executed successfully
  if $?.exitstatus != 0
    puts "Error: hipo-utils filter failed for #{cooked_OC_hipo}."
    next
  end

  # Step 3: Run recon-util-OC on _OC1.hipo
  puts "Running recon-util-OC on #{cooked_OC1_hipo}..."
  recon_util_command = "./tools/recon-util-OC -i #{cooked_OC1_hipo} -o #{cooked_OC2_hipo} -y ./recon/rga_fall2018_OC.yaml"
  system(recon_util_command)

  # Check if the recon-util-OC command executed successfully
  if $?.exitstatus != 0
    puts "Error: recon-util-OC failed for #{cooked_OC1_hipo}."
    next
  end

  # Step 4: Final filter to keep only specified banks
  puts "Applying final hipo-utils filter to keep only specific banks..."
  final_filter_command = "hipo-utils -filter -b 'RUN::*,MC::*,REC::Particle,REC::Calorimeter,REC::Track,REC::Traj,ECAL::*' #{cooked_OC2_hipo} -o #{final_filtered_hipo}"
  system(final_filter_command)

  # Check if the final filtering command executed successfully
  if $?.exitstatus != 0
    puts "Error: Final filtering failed for #{cooked_OC2_hipo}."
    next
  end

  puts "Processing complete for #{h5_file}. \n\n\nFinal HIPO file saved at #{final_filtered_hipo}"
end

puts "\t ==> All steps completed successfully for project #{project_name}!"
