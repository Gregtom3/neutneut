import pandas as pd
import matplotlib.pyplot as plt
from particle import Particle
plt.style.use('science')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})

# Function to get the LaTeX name of a particle by its PID
def get_particle_name(pid):
    if pid == -1:
        return "No match"
    try:
        particle = Particle.from_pdgid(pid)
        return f"${particle.latex_name}$"
    except Exception:
        return f"PID {pid}"

# Function to assign unique markers to particle types
def assign_marker(pid):
    marker_map = {
        11: 'o',  # electron
        2112: 's',  # neutron
        2212: '^',  # proton
        13: 'v',  # muon
        22: 'd',  # photon
        111: 'p',  # pi0
        211: 'H',  # pi+
        -211: 'h',  # pi-
        321: '*',  # K+
        -321: 'X',  # K-
        -1: 'P',  # Background
        -11: 'D'  # positron
    }
    return marker_map.get(pid, 'x')  # default to 'x' if PID not in map

def plot_ecal_peaks(csv_file, event_number):
    # Load the CSV data
    df = pd.read_csv(csv_file)
    
    # Filter data by the specified event number
    event_df = df[df['event_number'] == event_number]

    if event_df.empty:
        print(f"No data found for event number {event_number}")
        return

    # Define the colors
    colors = [
        (0.000, 0.000, 1.000),
        (0.000, 0.502, 0.000),
        (1.000, 0.000, 0.000),
        (0.000, 1.000, 1.000),
        (1.000, 0.000, 1.000),
        (1.000, 1.000, 0.000),
        (1.000, 0.647, 0.000),
        (0.502, 0.000, 0.502),
        (0.647, 0.165, 0.165),
        (0.275, 0.510, 0.706),
        (0.980, 0.502, 0.447),
        (0.824, 0.706, 0.549),
        (0.941, 0.902, 0.549),
        (0.502, 0.502, 0.502),
        (0.529, 0.808, 0.922),
        (0.941, 0.502, 0.502),
        (0.502, 0.000, 0.000),
        (0.184, 0.310, 0.310),
        (0.000, 0.749, 1.000)
    ]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

    # Left plot: Color by mcindex
    unique_mcindex = sorted(event_df['mc_index'].unique())
    color_map_mcindex = {mcindex: color for mcindex, color in zip(unique_mcindex, colors)}

    # Dictionary to track count of particles for unique labels
    mc_particle_count = {}

    for mcindex, color in color_map_mcindex.items():
        subset = event_df[event_df['mc_index'] == mcindex]
        mc_pid = subset['mc_pid'].values[0]
        mc_name = get_particle_name(mc_pid)
        marker = assign_marker(mc_pid)

        if mc_pid not in mc_particle_count:
            mc_particle_count[mc_pid] = 0
        mc_particle_count[mc_pid] += 1

        for _, row in subset.iterrows():
            ax1.plot([row['peak_xo'], row['peak_xe']], [row['peak_yo'], row['peak_ye']], color='black', alpha=0.3)
        if mc_pid != -1:
            ax1.scatter(subset['peak_xo'], subset['peak_yo'], label=f'MC {mc_name} ({mc_particle_count[mc_pid]})', color=color, marker=marker, s=100, edgecolor='k')
            ax1.scatter(subset['peak_xe'], subset['peak_ye'], color=color, marker=marker, s=100, edgecolor='k')
        else:
            ax1.scatter(subset['peak_xo'], subset['peak_yo'], label=f'MC {mc_name}', color="white", marker='X', s=100, edgecolor="black")
            ax1.scatter(subset['peak_xe'], subset['peak_ye'], color="white", marker='X', s=100, edgecolor="black")

    ax1.set_title('MC Generated')
    ax1.set_xlabel('ECAL::peaks (X)')
    ax1.set_ylabel('ECAL::peaks (Y)')
    ax1.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, fontsize=12)
    ax1.grid(True)

    # Right plot: Color by pindex
    unique_pindex = sorted(event_df['peak_pindex'].unique())
    color_map_pindex = {pindex: color for pindex, color in zip(unique_pindex, colors)}

    # Dictionary to track count of particles for unique labels
    rec_particle_count = {}

    for pindex, color in color_map_pindex.items():
        subset = event_df[event_df['peak_pindex'] == pindex]
        rec_pid = subset['peak_rec_pid'].values[0]
        rec_name = get_particle_name(rec_pid)
        marker = assign_marker(rec_pid)

        if rec_pid not in rec_particle_count:
            rec_particle_count[rec_pid] = 0
        rec_particle_count[rec_pid] += 1

        for _, row in subset.iterrows():
            ax2.plot([row['peak_xo'], row['peak_xe']], [row['peak_yo'], row['peak_ye']], color='black', alpha=0.3)
        if rec_pid != -1:
            ax2.scatter(subset['peak_xo'], subset['peak_yo'], label=f'REC {rec_name} ({rec_particle_count[rec_pid]})', color=color, marker=marker, s=100, edgecolor='k')
            ax2.scatter(subset['peak_xe'], subset['peak_ye'], color=color, marker=marker, s=100, edgecolor='k')
        else:
            ax2.scatter(subset['peak_xo'], subset['peak_yo'], label=f'REC {rec_name}', color="white", marker='X', s=100, edgecolor="black")
            ax2.scatter(subset['peak_xe'], subset['peak_ye'], color="white", marker='X', s=100, edgecolor="black")
            
    ax2.set_title('Reconstructed')
    ax2.set_xlabel('ECAL::peaks (X)')
    ax2.set_ylabel('ECAL::peaks (Y)')
    ax2.legend(frameon=True, loc='upper left', bbox_to_anchor=(0, -0.2), ncol=2, fontsize=12)
    ax2.grid(True)

    plt.show()

    
def plot_ecal_peaks_with_intersections(csv_file, intersections_csv, event_number):
    # Load the CSV data
    df = pd.read_csv(csv_file)
    intersections_df = pd.read_csv(intersections_csv)
    
    # Filter data by the specified event number
    event_df = df[df['event_number'] == event_number]
    intersection_event_df = intersections_df[intersections_df['event_number'] == event_number]

    if event_df.empty:
        print(f"No data found for event number {event_number}")
        return

    if intersection_event_df.empty:
        print(f"No intersections found for event number {event_number}")
        return

    # Define the colors
    colors = [
        (0.000, 0.000, 1.000),
        (0.000, 0.502, 0.000),
        (1.000, 0.000, 0.000),
        (0.000, 1.000, 1.000),
        (1.000, 0.000, 1.000),
        (1.000, 1.000, 0.000),
        (1.000, 0.647, 0.000),
        (0.502, 0.000, 0.502),
        (0.647, 0.165, 0.165),
        (0.275, 0.510, 0.706),
        (0.980, 0.502, 0.447),
        (0.824, 0.706, 0.549),
        (0.941, 0.902, 0.549),
        (0.502, 0.502, 0.502),
        (0.529, 0.808, 0.922),
        (0.941, 0.502, 0.502),
        (0.502, 0.000, 0.000),
        (0.184, 0.310, 0.310),
        (0.000, 0.749, 1.000)
    ]

    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # Plot peak lines
    for _, row in event_df.iterrows():
        ax.plot([row['peak_xo'], row['peak_xe']], [row['peak_yo'], row['peak_ye']], color='black', alpha=0.3)

    # Plot intersection points
    for _, row in intersection_event_df.iterrows():
        ax.scatter(row['x'], row['y'], color='orange', marker='o', s=25, edgecolor='k')

    ax.set_title(f'ECAL Peaks with Intersections for Event {event_number}')
    ax.set_xlabel('ECAL::peaks (X)')
    ax.set_ylabel('ECAL::peaks (Y)')
    plt.show()
    