import sys
import argparse
sys.path.append("/work/clas12/users/gmat/clas12/neutneut/src/")
from Evaluator import Evaluator
from ECALClusterAnalyzer import ECALClusterAnalyzer
from loss_functions import CustomLoss, AttractiveLossMetric, RepulsiveLossMetric, CowardLossMetric, NoiseLossMetric
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Run ECALClusterAnalyzer with user inputs.')
    
    parser.add_argument('--input_h5', type=str, required=True, help='Path to the input h5 file.')
    parser.add_argument('--original_hipofile', type=str, required=True, help='Path to the original hipo file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the tensorflow model.')
    parser.add_argument('--tB',type=float,required=True,help='Minimum beta for clustering.')
    parser.add_argument('--tD',type=float,required=True,help='Maximum distance for clustering.')
    parser.add_argument('--clustering_variable', type=str, default='otid', help='Clustering variable (default: "otid").')
    
    args = parser.parse_args()

    
    # Load the model
    model = tf.keras.models.load_model(args.model_path, custom_objects={
        'CustomLoss': CustomLoss, 
        'AttractiveLossMetric': AttractiveLossMetric,
        'RepulsiveLossMetric': RepulsiveLossMetric,
        'CowardLossMetric': CowardLossMetric,
        'NoiseLossMetric': NoiseLossMetric
    })
    
    # Setup Evaluator
    evaluator = Evaluator(args.input_h5)
    
    # Load model into Evaluator
    evaluator.load_model(model)
    
    # Pass h5 data into trained tensorflow model
    evaluator.predict()
    
    # Cluster object condensation variables
    evaluator.cluster(args.tB,args.tD)
    
    # Obtain cluster dataframe
    cluster_df = evaluator.to_cluster_dataframe()
    
    # Initialize ECALClusterAnalyzer with user inputs
    analyzer = ECALClusterAnalyzer(input_df = cluster_df, original_hipofile=args.original_hipofile, clustering_variable=args.clustering_variable)
    
    # Run the analyzer
    analyzer.run()
    
if __name__ == "__main__":
    main()
