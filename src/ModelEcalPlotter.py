import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from particle import Particle

class ModelEcalPlotter:
    
    def __init__(self, event_dataframe, colors=None):
        self.event = event_dataframe
        self.colors = colors if colors is not None else [
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
        self.object_ids = self.event["unique_mc_index"].values
        self.mc_pid = self.event["mc_pid"].values
        self.rec_pid = self.event["rec_pid"].values
        self.pindex = self.event["pindex"].values
        self.beta = self.event["beta"].values
        self.xc = self.event["xc"].values
        self.yc = self.event["yc"].values
        self.is_cluster_leader = self.event["is_cluster_leader"].values
        self.cluster_ids = self.event["cluster_id"].values
        self.event_xo = self.event["xo"].values
        self.event_yo = self.event["yo"].values
        self.event_xe = self.event["xe"].values
        self.event_ye = self.event["ye"].values
        self.N_hits = len(self.event)
        self.xc_range = (np.amin(self.xc) - 1, np.amax(self.xc) + 1)
        self.yc_range = (np.amin(self.yc) - 1, np.amax(self.yc) + 1)

    @staticmethod
    def get_particle_name(pid):
        if pid == -1:
            return "No match"
        try:
            particle = Particle.from_pdgid(pid)
            return f"${particle.latex_name}$"
        except Exception:
            return f"PID {pid}"

    @staticmethod
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

    @staticmethod
    def draw_six_sectors(ax):
        center = (0.5, 0.5)
        length = 2  # 2 units on each end
        angles = [-30, 30, 90]  # Angles in degrees
        for angle in angles:
            radians = np.radians(angle)
            start_x = center[0] - length * np.cos(radians)
            start_y = center[1] - length * np.sin(radians)
            end_x = center[0] + length * np.cos(radians)
            end_y = center[1] + length * np.sin(radians)
            ax.plot([start_x, end_x], [start_y, end_y], 'k:')

    def plot_mc_peaks(self, ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for iobj, object_id in enumerate(sorted(np.unique(self.object_ids))):
            idx = self.object_ids == object_id
            color = self.colors[iobj]
            pid = self.mc_pid[idx][0]
            marker = self.assign_marker(pid)
            name = self.get_particle_name(pid)
            ax.scatter(self.event_xo[idx], self.event_yo[idx], color=color, edgecolor="k", label=f"MC {name}", marker=marker)
            ax.scatter(self.event_xe[idx], self.event_ye[idx], color=color, edgecolor="k", marker=marker)
        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        for i in range(self.event_xo.shape[0]):
            ax.plot([self.event_xo[i], self.event_xe[i]], [self.event_yo[i], self.event_ye[i]], color='black', alpha=0.2)
        ax.set_xlabel("ECAL::peaks (X)")
        ax.set_ylabel("ECAL::peaks (Y)")
        self.draw_six_sectors(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def plot_latent_coordinates(self, ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for iobj, object_id in enumerate(sorted(np.unique(self.object_ids))):
            idx = self.object_ids == object_id
            color = self.colors[iobj]
            pid = self.mc_pid[idx][0]
            marker = self.assign_marker(pid)
            name = self.get_particle_name(pid)
            if object_id == -1:
                ax.scatter(self.xc[idx], self.yc[idx], edgecolor="k", color="white", label="Background")
            else:
                ax.scatter(self.xc[idx], self.yc[idx], edgecolor="k", color=color, label=f"MC {name}", marker=marker)
        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        ax.set_xlabel("Latent X Coordinate")
        ax.set_ylabel("Latent Y Coordinate")
        ax.set_xlim(self.xc_range[0], self.xc_range[1])
        ax.set_ylim(self.yc_range[0], self.yc_range[1])
        return ax

    def plot_rec_peaks(self, ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for iobj, object_id in enumerate(sorted(np.unique(self.pindex))):
            idx = self.pindex == object_id
            color = self.colors[iobj]
            pid = self.rec_pid[idx][0]
            marker = self.assign_marker(pid)
            name = self.get_particle_name(pid)
            ax.scatter(self.event_xo[idx], self.event_yo[idx], color=color, edgecolor="k", label=f"REC {name}", marker=marker)
            ax.scatter(self.event_xe[idx], self.event_ye[idx], color=color, edgecolor="k", marker=marker)
        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, 1), loc='lower center')
        for i in range(self.event_xo.shape[0]):
            ax.plot([self.event_xo[i], self.event_xe[i]], [self.event_yo[i], self.event_ye[i]], color='black', alpha=0.2)
        ax.set_xlabel("ECAL::peaks (X)")
        ax.set_ylabel("ECAL::peaks (Y)")
        self.draw_six_sectors(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def plot_clustered_ecal_peaks(self,ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = self.cluster_ids == cluster_id
            color = self.colors[-ic]
            if cluster_id == -1:
                ax.scatter(self.event_xo[idx], self.event_yo[idx], color="white", edgecolor="k", label=f"Background", marker="X")
                ax.scatter(self.event_xe[idx], self.event_ye[idx], color="white", edgecolor="k", marker="X")
            else:
                ax.scatter(self.event_xo[idx], self.event_yo[idx], color=color, edgecolor="k", label=f"Cluster {ic+1}")
                ax.scatter(self.event_xe[idx], self.event_ye[idx], color=color, edgecolor="k")
        for i in range(self.event_xo.shape[0]):
            ax.plot([self.event_xo[i], self.event_xe[i]], [self.event_yo[i], self.event_ye[i]], color='black', alpha=0.2, zorder=-1)
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            if cluster_id == -1:
                continue
            ihit = np.where((self.is_cluster_leader == 1) & (self.cluster_ids == cluster_id))[0][0]
            leader_xo, leader_yo = self.event_xo[ihit], self.event_yo[ihit]
            leader_xe, leader_ye = self.event_xe[ihit], self.event_ye[ihit]
            color = self.colors[-ic]
            ax.scatter([leader_xo, leader_xe], [leader_yo, leader_ye], color=color, s=150, edgecolor="k", hatch="...", marker="s")
        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, -0.15), loc='upper center')
        ax.set_xlabel("ECAL::peaks (X)")
        ax.set_ylabel("ECAL::peaks (Y)")
        self.draw_six_sectors(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def plot_cluster_latent_space(self, tD=None, ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        if tD == None:
            raise ValueError("Error: Must input valid tD.")
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = self.cluster_ids == cluster_id
            color = self.colors[-ic]
            if cluster_id == -1:
                ax.scatter(self.xc[idx], self.yc[idx], color="white", edgecolor="k", alpha=self.beta[idx])
                ax.scatter([], [], color="white", edgecolor="k", label=f"Background", alpha=1)
            else:
                ax.scatter(self.xc[idx], self.yc[idx], color=color, edgecolor="k", alpha=self.beta[idx])
                ax.scatter([], [], color=color, edgecolor="k", label=f"Cluster {ic+1}", alpha=1)
                for ihit in range(self.N_hits):
                    if self.is_cluster_leader[ihit] != 1 or self.cluster_ids[ihit] != cluster_id:
                        continue
                    leader_xc, leader_yc = self.xc[ihit], self.yc[ihit]
                    circle = patches.Circle((leader_xc, leader_yc), radius=tD, edgecolor=color, facecolor=color, alpha=0.1, hatch='//')
                    ax.add_patch(circle)
        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, -0.15), loc='upper center')
        ax.set_xlabel("Latent X Coordinate")
        ax.set_ylabel("Latent Y Coordinate")
        ax.set_xlim(self.xc_range[0], self.xc_range[1])
        ax.set_ylim(self.yc_range[0], self.yc_range[1])
        return ax

    def plot_beta_histogram(self, ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = self.cluster_ids == cluster_id
            color = self.colors[-ic]
            if cluster_id == -1:
                ax.hist(self.beta[idx], range=(0, 1), bins=25, edgecolor="black", zorder=10, histtype="step")
            else:
                ax.hist(self.beta[idx], range=(0, 1), bins=25, color=color, alpha=0.3)
        ax.set_xlabel("Learned $\\beta$")
        return ax

    def plot_all(self, tD):
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), dpi=150)
        axs = axs.flatten()
        self.plot_mc_peaks(ax=axs[0])
        self.plot_latent_coordinates(ax=axs[1])
        self.plot_rec_peaks(ax=axs[2])
        self.plot_clustered_ecal_peaks(ax=axs[3])
        self.plot_cluster_latent_space(tD, ax=axs[4])
        self.plot_beta_histogram(ax=axs[5])
        plt.tight_layout()
        return axs