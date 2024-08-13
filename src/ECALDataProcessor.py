import numpy as np

class ECALDataProcessor:
    """
    Class to process ECAL data by grouping hits and peaks by Monte Carlo index.
    """

    EPSILON = 0.0000001

    def group_hits_and_peaks_by_mcindex(self, ecal_hits, ecal_clusters, ecal_peaks, rec_calorimeter, mc_recmatch, rec_particle, mc_particle):
        """
        Group ECAL hits and peaks by Monte Carlo index.

        Parameters:
        -----------
        ecal_hits : pd.DataFrame
            DataFrame containing ECAL hits data.
        ecal_clusters : pd.DataFrame
            DataFrame containing ECAL clusters data.
        ecal_peaks : pd.DataFrame
            DataFrame containing ECAL peaks data.
        rec_calorimeter : pd.DataFrame
            DataFrame containing reconstructed calorimeter data.
        mc_recmatch : pd.DataFrame
            DataFrame containing Monte Carlo to reconstructed match data.
        rec_particle : pd.DataFrame
            DataFrame containing reconstructed particle data.
        mc_particle : pd.DataFrame
            DataFrame containing Monte Carlo particle data.

        Returns:
        --------
        dict
            Dictionary with grouped data by Monte Carlo index.
        """
        hits_grouped = ecal_hits.groupby(['peakid', 'clusterId'])
        grouped_data = {}

        for (peak_id, cluster_id), hits in hits_grouped:
            if cluster_id == -1:
                # Handle background hits where clusterId is -1
                if -1 not in grouped_data:
                    grouped_data[-1] = {
                        'hits': [],
                        'peaks': [],
                        'mc_pid': -1,
                        'rec_pids': {-1}
                    }

                # Assign hits and peaks to the background group
                self._assign_to_background(hits, ecal_peaks, grouped_data)
                continue
            
            # Process hits and peaks when clusterId is not -1
            self._process_non_background_hits(
                hits, ecal_clusters, ecal_peaks, rec_calorimeter, mc_recmatch,
                rec_particle, mc_particle, grouped_data, cluster_id
            )
    
        return grouped_data

    def _assign_to_background(self, hits, ecal_peaks, grouped_data):
        """
        Helper method to assign hits and peaks to the background group.

        Parameters:
        -----------
        hits : pd.DataFrame
            DataFrame containing the hits data.
        ecal_peaks : pd.DataFrame
            DataFrame containing the peaks data.
        grouped_data : dict
            Dictionary to store the grouped data.
        """
        hits_with_pindex = hits.copy()
        hits_with_pindex['pindex'] = -1  # Custom pindex for background
        grouped_data[-1]['hits'].extend(hits_with_pindex.to_dict('records'))

        peak_energy = hits['energy'].sum()
        matching_peaks = ecal_peaks[np.isclose(ecal_peaks['energy'], peak_energy, atol=self.EPSILON)]
        for _, peak in matching_peaks.iterrows():
            peak_with_pindex = peak.copy()
            peak_with_pindex['pindex'] = -1  # Custom pindex for background
            peak_with_pindex['rec_pid'] = -1  # Custom pid for background
            grouped_data[-1]['peaks'].append(peak_with_pindex.to_dict())

    def _process_non_background_hits(self, hits, ecal_clusters, ecal_peaks, rec_calorimeter, mc_recmatch, rec_particle, mc_particle, grouped_data, cluster_id):
        """
        Helper method to process non-background hits and peaks.

        Parameters:
        -----------
        hits : pd.DataFrame
            DataFrame containing the hits data.
        ecal_clusters : pd.DataFrame
            DataFrame containing the clusters data.
        ecal_peaks : pd.DataFrame
            DataFrame containing the peaks data.
        rec_calorimeter : pd.DataFrame
            DataFrame containing the reconstructed calorimeter data.
        mc_recmatch : pd.DataFrame
            DataFrame containing the Monte Carlo to reconstructed match data.
        rec_particle : pd.DataFrame
            DataFrame containing the reconstructed particle data.
        mc_particle : pd.DataFrame
            DataFrame containing the Monte Carlo particle data.
        grouped_data : dict
            Dictionary to store the grouped data.
        cluster_id : int
            The cluster ID of the hits.
        """
        cluster_energy = ecal_clusters['energy'][cluster_id - 1]
        rec_cal_row = rec_calorimeter[np.isclose(rec_calorimeter['energy'], cluster_energy, atol=self.EPSILON)]
        if not rec_cal_row.empty:
            pindex = rec_cal_row['pindex'].values[0]
            mcindex = mc_recmatch[mc_recmatch['pindex'] == pindex]['mcindex'].values[0]

            if mcindex not in grouped_data:
                grouped_data[mcindex] = {
                    'hits': [],
                    'peaks': [],
                    'mc_pid': None,
                    'rec_pids': set()
                }

            # Add hits and peaks to the mcindex entry
            self._add_hits_and_peaks_to_group(hits, ecal_peaks, rec_particle, mc_particle, grouped_data, pindex, mcindex)
            
    def _add_hits_and_peaks_to_group(self, hits, ecal_peaks, rec_particle, mc_particle, grouped_data, pindex, mcindex):
        """
        Helper method to add hits and peaks to the grouped data.

        Parameters:
        -----------
        hits : pd.DataFrame
            DataFrame containing the hits data.
        ecal_peaks : pd.DataFrame
            DataFrame containing the peaks data.
        rec_particle : pd.DataFrame
            DataFrame containing the reconstructed particle data.
        mc_particle : pd.DataFrame
            DataFrame containing the Monte Carlo particle data.
        grouped_data : dict
            Dictionary to store the grouped data.
        pindex : int
            The particle index.
        mcindex : int
            The Monte Carlo index.
        """
        hits_with_pindex = hits.copy()
        hits_with_pindex['pindex'] = pindex
        grouped_data[mcindex]['hits'].extend(hits_with_pindex.to_dict('records'))

        peak_energy = hits['energy'].sum()
        matching_peaks = ecal_peaks[np.isclose(ecal_peaks['energy'], peak_energy, atol=self.EPSILON)]
        
        if len(matching_peaks) > 1:
            # Check if 80% or more hits are in the same sector
            sector_counts = hits['sector'].value_counts(normalize=True)
            dominant_sector = sector_counts.idxmax()
            if sector_counts[dominant_sector] >= 0.8:
                # Match to peak with same sector
                matching_peaks = matching_peaks[matching_peaks['sector'] == dominant_sector]

        if matching_peaks.empty:
            # Assign hits to background if no matching peak found
            self._assign_to_background(hits, ecal_peaks, grouped_data)
        else:
            # Assign matched peaks to group
            for _, peak in matching_peaks.iterrows():
                peak_with_pindex = peak.copy()
                peak_with_pindex['pindex'] = int(pindex)
                rec_particle_row = rec_particle.iloc[pindex]
                peak_with_pindex['rec_pid'] = rec_particle_row['pid']
                grouped_data[mcindex]['peaks'].append(peak_with_pindex.to_dict())

            mc_particle_row = mc_particle.iloc[mcindex]
            grouped_data[mcindex]['mc_pid'] = mc_particle_row['pid']
            rec_particle_row = rec_particle.iloc[pindex]
            grouped_data[mcindex]['rec_pids'].add(rec_particle_row['pid'])
