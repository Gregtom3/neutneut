import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from particle import Particle
from global_params import *
import pandas as pd

class ModelEcalPlotter:
    
    def __init__(self, event_dataframe, colors=None, use_clas_calo_scale=False):
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
        self.object_ids = self.event["unique_otid"].values
        self.mc_pid = self.event["mc_pid"].values
        self.rec_pid = self.event["rec_pid"].values
        self.pindex = self.event["pindex"].values
        self.beta = self.event["beta"].values
        self.xc = self.event["xc"].values
        self.yc = self.event["yc"].values
        if 'centroid_x' not in self.event.columns:
            self.centroid_x = self.event["x"].values
            self.centroid_y = self.event["y"].values
        else:
            self.centroid_x = self.event["centroid_x"].values
            self.centroid_y = self.event["centroid_y"].values
        if 'cluster_x' in self.event.columns:
            self.has_reco_cluster = True
            self.reco_cluster_x = self.event["cluster_x"].values
            self.reco_cluster_y = self.event["cluster_y"].values
        else:
            self.has_reco_cluster = False
        self.layer = self.event["layer"].values
        self.is_cluster_leader = self.event["is_cluster_leader"].values
        self.cluster_ids = self.event["cluster_id"].values
        self.pred_pid    = self.event["pred_pid"].values
        self.event_xo = self.event["xo"].values
        self.event_yo = self.event["yo"].values
        self.event_xe = self.event["xe"].values
        self.event_ye = self.event["ye"].values
        self.N_hits = len(self.event)
        self.xc_range = (np.amin(self.xc) - 1, np.amax(self.xc) + 1)
        self.yc_range = (np.amin(self.yc) - 1, np.amax(self.yc) + 1)
        self.use_clas_calo_scale = use_clas_calo_scale

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
        }
        return marker_map.get(pid, 'D')  # default to 'D' if PID not in map

    def draw_six_sectors(self,ax):
        center = (0.5, 0.5)
        length = 4 if not self.use_clas_calo_scale else 1000
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
            ax.plot([self.event_xo[i], self.event_xe[i]], [self.event_yo[i], self.event_ye[i]], color='black', alpha=0.075)
        for (x,y,l) in zip(self.centroid_x,self.centroid_y,self.layer):
            if x==0 and y==0:
                continue
            # Commenting out centroid plotting
            # if l in [1,2,3]:
            #     ax.scatter(x,y,s=25,color="gray",marker="o",edgecolor="k",zorder=100)
            # elif l in [4,5,6]:
            #     ax.scatter(x,y,s=25,color="gray",marker="^",edgecolor="k",zorder=100)
            # elif l in [7,8,9]:
            #     ax.scatter(x,y,s=25,color="gray",marker="v",edgecolor="k",zorder=100)
        ax.set_xlabel("ECAL::hits (X)")
        ax.set_ylabel("ECAL::hits (Y)")
        self.draw_six_sectors(ax)
        if self.use_clas_calo_scale:
            ax.set_xlim(ECAL_xy_min, ECAL_xy_max)
            ax.set_ylim(ECAL_xy_min, ECAL_xy_max)
        else:
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
        return ax

    def plot_latent_coordinates(self, ax=None, latent_space_lims=None):
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

        if latent_space_lims!=None:
            ax.set_xlim(latent_space_lims[0][0],latent_space_lims[0][1])
            ax.set_ylim(latent_space_lims[1][0],latent_space_lims[1][1])

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
        ax.set_xlabel("ECAL::hits (X)")
        ax.set_ylabel("ECAL::hits (Y)")
        self.draw_six_sectors(ax)
        if self.use_clas_calo_scale:
            ax.set_xlim(ECAL_xy_min, ECAL_xy_max)
            ax.set_ylim(ECAL_xy_min, ECAL_xy_max)
        else:
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
        return ax


    def plot_clustered_ecal_peaks(self,ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = self.cluster_ids == cluster_id
            color = self.colors[ic % len(self.colors)]
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
            leader_pid_type = self.pred_pid[ihit]
            color = self.colors[ic % len(self.colors)]
            # Choose marker based on leader_pid_type
            if leader_pid_type == 0:
                marker = 's'  # Diamond
            elif leader_pid_type == 1:
                marker = 's'  # Square
            elif leader_pid_type == 2:
                marker = 's'  # Circle
            else:
                marker = 's'  # Fallback marker in case of an unexpected pid_type
            
            # Scatter plot with dynamic marker
            ax.scatter([leader_xo, leader_xe], [leader_yo, leader_ye], 
                       color=color, s=150, edgecolor="k", hatch="...", marker=marker)
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            if cluster_id == -1:
                continue
            ihit = np.where((self.cluster_ids == cluster_id))[0][0]
            if self.has_reco_cluster==True:
                cluster_x = self.reco_cluster_x[ihit]
                cluster_y = self.reco_cluster_y[ihit]
                color = self.colors[ic % len(self.colors)]
                # Commenting out centroid plotting
                # ax.scatter(cluster_x,cluster_y,color=color,s=15,edgecolor="k",marker="o",zorder=100)
        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, -0.15), loc='upper center')
        ax.set_xlabel("ECAL::hits (X)")
        ax.set_ylabel("ECAL::hits (Y)")
        self.draw_six_sectors(ax)
        if self.use_clas_calo_scale:
            ax.set_xlim(ECAL_xy_min, ECAL_xy_max)
            ax.set_ylim(ECAL_xy_min, ECAL_xy_max)
        else:
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
        return ax

    def plot_cluster_latent_space(self, tD=None, ax=None, latent_space_lims=None):
        if ax==None:
            fig, ax = plt.subplots()
        if tD == None:
            raise ValueError("Error: Must input valid tD.")
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = self.cluster_ids == cluster_id
            color = self.colors[ic % len(self.colors)]
            if cluster_id == -1:
                ax.scatter(self.xc[idx], self.yc[idx], color="white", edgecolor="k", alpha=np.maximum(self.beta[idx],0.05))
                ax.scatter([], [], color="white", edgecolor="k", label=f"Background", alpha=1)
            else:
                ax.scatter(self.xc[idx], self.yc[idx], color=color, edgecolor="k", alpha=np.maximum(self.beta[idx],0.05))
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
        if latent_space_lims!=None:
            ax.set_xlim(latent_space_lims[0][0],latent_space_lims[0][1])
            ax.set_ylim(latent_space_lims[1][0],latent_space_lims[1][1])
        return ax

    def plot_beta_histogram(self, ax=None):
        if ax==None:
            fig, ax = plt.subplots()
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = self.cluster_ids == cluster_id
            color = self.colors[ic % len(self.colors)]
            if cluster_id == -1:
                continue # Do not histogram the noise
                ax.hist(self.beta[idx], range=(0, 1), bins=100, edgecolor="black", zorder=10, histtype="step")
            else:
                ax.hist(self.beta[idx], range=(0, 1), bins=100, color=color, alpha=0.3)
        ax.set_xlabel("Learned $\\beta$")
        return ax

    
    def plot_clustered_ecal_peaks_v2(self, ax=None, layergroup=[1, 2, 3]):
        if ax is None:
            fig, ax = plt.subplots()

        idx_in_layer_group = np.array([layer in layergroup for layer in self.layer])

        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            idx = (self.cluster_ids == cluster_id) & idx_in_layer_group
            color = self.colors[ic % len(self.colors)]

            if cluster_id == -1:
                ax.scatter(self.event_xo[idx], self.event_yo[idx], color="white", edgecolor="k", label="Background", marker="X")
                ax.scatter(self.event_xe[idx], self.event_ye[idx], color="white", edgecolor="k", marker="X")
            else:
                ax.scatter(self.event_xo[idx], self.event_yo[idx], color=color, edgecolor="k", label=f"Cluster {ic + 1}")
                ax.scatter(self.event_xe[idx], self.event_ye[idx], color=color, edgecolor="k")

        # Draw lines between points, applying layer group filter
        for i in range(self.event_xo.shape[0]):
            if idx_in_layer_group[i]:
                ax.plot([self.event_xo[i], self.event_xe[i]], [self.event_yo[i], self.event_ye[i]], color='black', alpha=0.2, zorder=-1)

        # Plot cluster leaders
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            if cluster_id == -1:
                continue
            ihit = np.where((self.is_cluster_leader == 1) & (self.cluster_ids == cluster_id) & idx_in_layer_group)[0]
            if ihit.size > 0:
                leader_idx = ihit[0]
                leader_xo, leader_yo = self.event_xo[leader_idx], self.event_yo[leader_idx]
                leader_xe, leader_ye = self.event_xe[leader_idx], self.event_ye[leader_idx]
                color = self.colors[ic % len(self.colors)]
                ax.scatter([leader_xo, leader_xe], [leader_yo, leader_ye], color=color, s=150, edgecolor="k", hatch="...", marker="s")

        # Plot reconstructed clusters
        for ic, cluster_id in enumerate(sorted(np.unique(self.cluster_ids))):
            if cluster_id == -1:
                continue
            ihit = np.where((self.cluster_ids == cluster_id) & idx_in_layer_group)[0]
            if ihit.size > 0 and self.has_reco_cluster:
                cluster_idx = ihit[0]
                cluster_x = self.reco_cluster_x[cluster_idx]
                cluster_y = self.reco_cluster_y[cluster_idx]
                color = self.colors[ic % len(self.colors)]
                ax.scatter(cluster_x, cluster_y, color=color, s=30, edgecolor="k", marker="x")

        ax.legend(frameon=True, ncols=2, bbox_to_anchor=(0.5, -0.15), loc='upper center')
        ax.set_xlabel("ECAL::hits (X)")
        ax.set_ylabel("ECAL::hits (Y)")
        self.draw_six_sectors(ax)

        if self.use_clas_calo_scale:
            ax.set_xlim(ECAL_xy_min, ECAL_xy_max)
            ax.set_ylim(ECAL_xy_min, ECAL_xy_max)
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        return ax
    

    def plot_all(self, tD, out=None, suptitle=None, latent_space_lims=None):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=150, facecolor='white')
        axs = axs.flatten()
        self.plot_mc_peaks(ax=axs[0])
        self.plot_latent_coordinates(ax=axs[1], latent_space_lims=latent_space_lims)
        self.plot_rec_peaks(ax=axs[2])
        self.plot_clustered_ecal_peaks(ax=axs[3])
        self.plot_cluster_latent_space(tD, ax=axs[4], latent_space_lims=latent_space_lims)
        self.plot_beta_histogram(ax=axs[5])
        
        # Adjust the spacing between subplots to make them slightly smaller
        plt.subplots_adjust(left=0.08, right=0.92, top=0.825, bottom=0.15, wspace=0.3, hspace=0.3)
    
        if suptitle!=None:
            fig.suptitle(suptitle,fontsize=20)
        if out!=None:
            plt.savefig(out)
            plt.close()
        else:
            return fig,axs