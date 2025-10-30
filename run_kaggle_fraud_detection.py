"""
Script to run fraud detection on your Kaggle dataset
Replace 'your_dataset.csv' with the path to your downloaded Kaggle dataset
"""

from fraud_detection_model import FraudDetectionModel
import pandas as pd

def run_fraud_detection():
    """Run fraud detection on your dataset."""
    
    # Initialize the model
    fraud_model = FraudDetectionModel()
    
    # Load your Kaggle dataset
    # Using the actual dataset path
    dataset_path = 'fraudTest.csv/fraudTest.csv'  # Updated to use the actual dataset
    
    print("Loading your Kaggle fraud detection dataset...")
    
    try:
        # Load the dataset
        if fraud_model.load_data(dataset_path):
            print("Dataset loaded successfully!")
            
            # Explore the data
            fraud_model.explore_data()
            
            # Preprocess the data
            # The script will auto-detect common target column names
            # If your target column has a different name, specify it here
            fraud_model.preprocess_data()  # Auto-detect target column
            
            # Split the data
            fraud_model.split_data()
            
            # Handle class imbalance
            fraud_model.handle_imbalance('smote')
            
            # Train models
            fraud_model.train_models()
            
            # Evaluate models
            results = fraud_model.evaluate_models()
            
            # Save the best model
            fraud_model.save_model('kaggle_fraud_model.pkl')
            
            print("\n" + "="*60)
            print("FRAUD DETECTION MODEL TRAINING COMPLETED!")
            print("="*60)
            print("Your trained model has been saved as 'kaggle_fraud_model.pkl'")
            print("You can now use this model to detect fraud in new transactions!")
            
            # Show which model performed best
            best_model_name = max(results.keys(), key=lambda x: results[x]['auc_score'])
            print(f"\nBest performing model: {best_model_name}")
            print(f"Best AUC Score: {results[best_model_name]['auc_score']:.4f}")
            
            # Generate plots
            print("\nGenerating evaluation plots...")
            fraud_model.plot_results()
            
            # Show feature importance
            print("\nAnalyzing feature importance...")
            fraud_model.get_feature_importance()
            
            print("\n" + "="*60)
            print("ADDITIONAL ANALYSIS COMPLETED!")
            print("="*60)
            print("Check the generated 'fraud_detection_results.png' for visualizations")
            print("The model is ready for production use!")
            
        else:
            print("Failed to load dataset. Please check the file path.")
            
    except FileNotFoundError:
        print(f"Dataset file '{dataset_path}' not found.")
        print("Please ensure the fraud detection dataset is in the correct location.")
        print("The dataset should be located at: fraudTest.csv/fraudTest.csv")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your dataset format and try again.")

if __name__ == "__main__":
    run_fraud_detection()
