package core;

import java.util.HashMap;
import java.util.Map;

public class MultiClassConfusionMatrix {
	
	public int num_classes ; 
	public int [][] confusion_matrix ; 
	
	public MultiClassConfusionMatrix(int [] true_labels, int [] predicted_labels, int numclasses)
	{
		this.num_classes = numclasses ; 
		this.confusion_matrix = new int[numclasses][numclasses] ; 
		for (int index = 0 ; index < true_labels.length; index++) {
			confusion_matrix[true_labels[index] - 1][predicted_labels[index] - 1]++ ; 
		}
	}
	public double RecallAtClass(int classIndex) {
        int truePositives = confusion_matrix[classIndex][classIndex];
        int falseNegatives = 0;

        for (int i = 0; i < this.num_classes; i++) {
            if (i != classIndex) {
                falseNegatives += confusion_matrix[classIndex][i];
            }
        }

        return (double) truePositives / (truePositives + falseNegatives);
    }
	
	public double PrecisionAtClass(int classIndex) {
        int truePositives = confusion_matrix[classIndex][classIndex];
        int falsePositives = 0;

        for (int i = 0; i < num_classes; i++) {
            if (i != classIndex) {
                falsePositives += confusion_matrix[i][classIndex];
            }
        }

        return (double) truePositives / (truePositives + falsePositives);
    }
	
	public double f1Score(int classIndex) {
        double precision = PrecisionAtClass(classIndex);
        double recall = RecallAtClass(classIndex);

        return 2 * (precision * recall) / (precision + recall);
    }
	
	public double Gmean() {
        double res = 1.0 ; 
        for (int i = 0 ; i < this.num_classes ; i++) {
        	res *= RecallAtClass(i) ; 
        }
        return Math.sqrt(res) ; 
    }
	public Map<String, Object> get_all_stats() {
		Map<String, Object> statistics = new HashMap<String, Object>() ; 
		
		for (int iclass = 0 ; iclass < this.num_classes ; iclass++) {
			statistics.put("Recall_"+iclass, String.valueOf(this.RecallAtClass(iclass))) ; 
			statistics.put("Precision_"+iclass, String.valueOf(this.PrecisionAtClass(iclass))) ; 
			statistics.put("F1_"+iclass, String.valueOf(this.f1Score(iclass))) ; 
		}
		statistics.put("Gmean", String.valueOf(this.Gmean())) ; 
		
		return statistics;
		
	}
	
}
