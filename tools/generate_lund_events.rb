#!/usr/bin/env ruby
require 'optparse'
require 'fileutils'
require 'securerandom'

# Default values
default_values = {
  beam_p: "proton, 4.0*GeV, 20.0*deg, 10*deg",
  spread_p: "1*GeV, 10*deg, 180*deg",
  mode: "e+n"
}

# Parse arguments
options = {}
OptionParser.new do |opts|
  opts.banner = "Usage: generate_lund_events.rb [options]"

  opts.on("--output OUTPUT", "Output directory") do |v|
    options[:output] = v
  end
    
  opts.on("--nevents NEVENTS", "Number of Events") do |v|
    options[:n] = v.to_i
  end
    
  opts.on("--beam_p BEAM_P", "Beam momentum") do |v|
    options[:beam_p] = v
  end
    
  opts.on("--mode MODE", "Mode (e+n or p+n or e+p+n or e+n?)") do |v|
    if ["e+n", "p+n", "e+p+n", "e+n?"].include?(v)
      options[:mode] = v
    else
      puts "Invalid mode. Allowed values are 'e+n', 'p+n', 'e+p+n', or 'e+n?'."
      exit
    end
  end
    
  opts.on("--spread_p SPREAD_P", "Spread momentum") do |v|
    options[:spread_p] = v
  end
end.parse!

# Check required options
[:output, :n].each do |opt|
  if options[opt].nil?
    puts "Missing required argument: --#{opt}"
    exit
  end
end

# Apply default values if not provided
default_values.each do |key, value|
  options[key] ||= value
end

# Helper function to generate random value within the spread
def random_sample(central, spread)
  central + spread * (SecureRandom.random_number - 0.5) * 2
end

# Convert degrees to radians
def deg_to_rad(deg)
  deg * Math::PI / 180
end

# Parse beam and spread values
particle_name = options[:beam_p].split(",").first.strip
central_p = options[:beam_p].split(",")[1].strip.gsub("*GeV", "").to_f
central_theta = deg_to_rad(options[:beam_p].split(",")[2].strip.gsub("*deg", "").to_f)
central_phi = deg_to_rad(options[:beam_p].split(",")[3].strip.gsub("*deg", "").to_f)

spread_p = options[:spread_p].split(",")[0].strip.gsub("*GeV", "").to_f
spread_theta = deg_to_rad(options[:spread_p].split(",")[1].strip.gsub("*deg", "").to_f)
spread_phi = deg_to_rad(options[:spread_p].split(",")[2].strip.gsub("*deg", "").to_f)

# Generate the output file name and path
params = [
  options[:beam_p].split(",").map(&:strip).map { |v| v.gsub(/[^0-9.]/, '') },
  options[:spread_p].split(",").map(&:strip).map { |v| v.gsub(/[^0-9.]/, '') },
  particle_name
].flatten.join("___")

output_filename = "#{params}___.lund"
output_path = File.join(options[:output], output_filename)

# Create output directory if it doesn't exist
FileUtils.mkdir_p(options[:output])

# Generate 'n' events
File.open(output_path, 'w') do |file|
  options[:n].times do
    # Randomly sample Px, Py, Pz
    sampled_p = random_sample(central_p, spread_p)
    sampled_theta = random_sample(central_theta, spread_theta)
    sampled_phi = random_sample(central_phi, spread_phi)

    # Calculate neutron Px, Py, Pz
    n_px = (sampled_p * Math.sin(sampled_theta) * Math.cos(sampled_phi)).round(6)
    n_py = (sampled_p * Math.sin(sampled_theta) * Math.sin(sampled_phi)).round(6)
    n_pz = (sampled_p * Math.cos(sampled_theta)).round(6)

    if options[:mode] == "e+n"
      pT = Math.sqrt(n_px * n_px + n_py * n_py)
      pT_scale = 1
      e_px = -(pT_scale * n_px / pT).round(6)
      e_py = -(pT_scale * n_py / pT).round(6)
      e_pz = 3

      # Calculate E and Ee
      mass_proton = 0.939565
      mass_electron = 0.000511
      e = Math.sqrt(n_px**2 + n_py**2 + n_pz**2 + mass_proton**2).round(6)
      ee = Math.sqrt(e_px**2 + e_py**2 + e_pz**2 + mass_electron**2).round(6)

      # Write the 3 lines for each event
      file.puts "2 1 1 0 0 11 10.600000 2212 0 1.000000"
      file.puts "1 1 1 11 0 0 #{e_px} #{e_py} #{e_pz} #{ee} 0.000511 0.000000 0.000000 0.000000"
      file.puts "2 1 1 2112 0 0 #{n_px} #{n_py} #{n_pz} #{e} 0.939565 0.000000 0.000000 0.000000"
    elsif options[:mode] == "p+n"
      # Write the 3 lines for each event
      file.puts "2 1 1 0 0 11 10.600000 2212 0 1.000000"
      file.puts "1 1 1 2212 0 0 -#{n_px} -#{n_px} #{n_pz} #{e} 0.939565 0.000000 0.000000 0.000000"
      file.puts "2 1 1 2112 0 0 #{n_px} #{n_py} #{n_pz} #{e} 0.939565 0.000000 0.000000 0.000000"
    elsif options[:mode] == "e+p+n"
      # Proton with 120 degrees offset in phi
      p_phi = sampled_phi + (120 * Math::PI / 180)
      p_px = (sampled_p * Math.sin(sampled_theta) * Math.cos(p_phi)).round(6)
      p_py = (sampled_p * Math.sin(sampled_theta) * Math.sin(p_phi)).round(6)
      p_pz = n_pz

      # Electron with 240 degrees offset in phi
      e_phi = sampled_phi + (240 * Math::PI / 180)
      e_px = (4 * Math.sin(15 * Math::PI / 180) * Math.cos(e_phi)).round(6)
      e_py = (4 * Math.sin(15 * Math::PI / 180) * Math.sin(e_phi)).round(6)
      e_pz = (4 * Math.cos(15 * Math::PI / 180)).round(6)

      # Calculate energies
      mass_proton = 0.939565
      mass_electron = 0.000511
      e_n = Math.sqrt(n_px**2 + n_py**2 + n_pz**2 + mass_proton**2).round(6)
      e_p = Math.sqrt(p_px**2 + p_py**2 + p_pz**2 + mass_proton**2).round(6)
      e_e = Math.sqrt(e_px**2 + e_py**2 + e_pz**2 + mass_electron**2).round(6)

      # Write the 3 lines for each event
      file.puts "3 1 1 0 0 11 10.600000 2212 0 1.000000"
      file.puts "1 1 1 11 0 0 #{e_px} #{e_py} #{e_pz} #{e_e} 0.000511 0.000000 0.000000 0.000000"
      file.puts "2 1 1 2212 0 0 #{p_px} #{p_py} #{p_pz} #{e_p} 0.939565 0.000000 0.000000 0.000000"
      file.puts "3 1 1 2112 0 0 #{n_px} #{n_py} #{n_pz} #{e_n} 0.939565 0.000000 0.000000 0.000000"
    elsif options[:mode] == "e+n?"
      # Electron's parameters remain constant in all cases
      pT = Math.sqrt(n_px * n_px + n_py * n_py)
      pT_scale = 1
      e_px = -(pT_scale * n_px / pT).round(6)
      e_py = -(pT_scale * n_py / pT).round(6)
      e_pz = 3

      # Calculate Ee
      mass_electron = 0.000511
      ee = Math.sqrt(e_px**2 + e_py**2 + e_pz**2 + mass_electron**2).round(6)

      if SecureRandom.random_number < 0.5
        # Generate as "e+n" mode
        mass_proton = 0.939565
        e = Math.sqrt(n_px**2 + n_py**2 + n_pz**2 + mass_proton**2).round(6)

        # Write the 3 lines for each event
        file.puts "2 1 1 0 0 11 10.600000 2212 0 1.000000"
        file.puts "1 1 1 11 0 0 #{e_px} #{e_py} #{e_pz} #{ee} 0.000511 0.000000 0.000000 0.000000"
        file.puts "2 1 1 2112 0 0 #{n_px} #{n_py} #{n_pz} #{e} 0.939565 0.000000 0.000000 0.000000"
      else
        # Add extra neutrons
        num_extra_neutrons = [1, 2].sample
        extra_neutrons = []

        num_extra_neutrons.times do
          delta_theta = random_sample(0, 0.1)  # Deviation within 0.1 rad
          delta_phi = random_sample(0, 0.1)    # Deviation within 0.1 rad
          n_p_extra = random_sample(sampled_p, sampled_p * 0.05)

          extra_theta = sampled_theta + delta_theta
          extra_phi = sampled_phi + delta_phi

          extra_px = (n_p_extra * Math.sin(extra_theta) * Math.cos(extra_phi)).round(6)
          extra_py = (n_p_extra * Math.sin(extra_theta) * Math.sin(extra_phi)).round(6)
          extra_pz = (n_p_extra * Math.cos(extra_theta)).round(6)

          extra_neutrons << [extra_px, extra_py, extra_pz]
        end

        # Write the lines for each event including the electron and neutrons
        mass_proton = 0.939565
        e = Math.sqrt(n_px**2 + n_py**2 + n_pz**2 + mass_proton**2).round(6)
        file.puts "#{2 + num_extra_neutrons} 1 1 0 0 11 10.600000 2212 0 1.000000"
        file.puts "1 1 1 11 0 0 #{e_px} #{e_py} #{e_pz} #{ee} 0.000511 0.000000 0.000000 0.000000"
        file.puts "2 1 1 2112 0 0 #{n_px} #{n_py} #{n_pz} #{e} 0.939565 0.000000 0.000000 0.000000"

        extra_neutrons.each_with_index do |(px, py, pz), index|
          e_extra = Math.sqrt(px**2 + py**2 + pz**2 + mass_proton**2).round(6)
          file.puts "#{index + 3} 1 1 2112 0 0 #{px} #{py} #{pz} #{e_extra} 0.939565 0.000000 0.000000 0.000000"
        end
      end
    end
  end
end

puts "Generated lund file: #{output_path}"
