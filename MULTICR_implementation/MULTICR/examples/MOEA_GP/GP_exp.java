package MOEA_GP;

import org.moeaframework.Executor;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.Solution;
import org.moeaframework.util.tree.Node;

import core.ConfusionMatrix;
import core.CsvDataFrame;
import core.DataFrame;
import core.TreeView;
import core.GraphViz;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

public class GP_exp {
	private NondominatedPopulation solutions ; 
	private  ArrayList<ConfusionMatrix> train_confusion_matrices ; 
	private  ArrayList<ConfusionMatrix> test_confusion_matrices ; 
	private HashMap<String,String> run_configs ; 
	public GP_exp(DataFrame train_data,DataFrame test_data, HashMap<String,Integer> features,String topredict,String algo,String[] objectives,int pop_size,int max_eval,Double mut_prob,Double cross_prob) {
		
		//saving run parameters 
		this.run_configs = new HashMap<String,String>() ; 
		
		this.run_configs.put("algorithm", algo); 
		this.run_configs.put("populationsize",  String.valueOf(pop_size)); 
		this.run_configs.put("mutation_rate",  String.valueOf(mut_prob)); 
		this.run_configs.put("crossover_rate",  String.valueOf(cross_prob)); 
		this.run_configs.put("max_evaluation_number",  String.valueOf(max_eval)); 
		
		
		
		
		this.run_configs.put("algorithm", algo); 
		Object[] true_train_labels = train_data.get_column_data(topredict) ; 
		Object[] true_test_labels = test_data.get_column_data(topredict) ; 
		
		this.solutions = new Executor()
			      .withProblemClass(Train_MOGP.class, train_data,true_train_labels,features,objectives)
			      .withAlgorithm(algo)
			      .withProperty("populationSize", pop_size)
			      .withProperty("operator", "bx")
			      .withProperty("operator", "ptm")
			      .withProperty("bx.rate",cross_prob)
			      .withProperty("ptm.rate", mut_prob)
			      .distributeOnAllCores()
			      .withMaxEvaluations(max_eval)
			      .run();  
		
		this.train_confusion_matrices = new  ArrayList<ConfusionMatrix>() ; 
		this.test_confusion_matrices = new  ArrayList<ConfusionMatrix>() ; 
		
		
		for (Solution sol : this.solutions)
		{
			 Boolean [] train_predictions = Gp_try.compute_predictions(sol,train_data.getData(),features.keySet()) ; 
			 Boolean [] test_predictions = Gp_try.compute_predictions(sol,test_data.getData(),features.keySet()) ; 
			 
			 ConfusionMatrix train_confusion_matrix = new ConfusionMatrix(true_train_labels,train_predictions) ;
			 ConfusionMatrix test_confusion_matrix = new ConfusionMatrix(true_test_labels,test_predictions) ;
			 
			 train_confusion_matrices.add(train_confusion_matrix) ; 
			 test_confusion_matrices.add(test_confusion_matrix) ; 
			 
			 
		}
	
		
	}
	public NondominatedPopulation getSolutions()
	{
		return this.solutions ; 
	}
	public ArrayList<ConfusionMatrix> getrain_confusion_matrices()
	{
		return this.train_confusion_matrices ; 
	}
	public ArrayList<ConfusionMatrix> getest_confusion_matrices()
	{
		return this.test_confusion_matrices ; 
	}
	public void save_data(String path)
	{
		try {
			//System.out.println("lol" + path);
			ObjectOutputStream train_confusion_matrix = new ObjectOutputStream(new FileOutputStream(path+"/train_confusion_matrix.txt"));
			ObjectOutputStream test_confusion_matrix = new ObjectOutputStream(new FileOutputStream(path+"/test_confusion_matrix.txt"));
			ObjectOutputStream parameters = new ObjectOutputStream(new FileOutputStream(path+"/parameters.txt"));
			
			int i = 0 ; 
			for (ConfusionMatrix m : this.train_confusion_matrices) 
			{
				train_confusion_matrix.writeChars(m.get_statistics().toString());
				i++ ; 
			}
			i = 0 ; 
			train_confusion_matrix.close(); 
			for (ConfusionMatrix m : this.test_confusion_matrices) 
			{
				test_confusion_matrix.writeChars(m.get_statistics().toString());
				i++ ; 
			}
			test_confusion_matrix.close(); 
			parameters.writeChars(this.run_configs.toString());
			parameters.close();
			i = 0 ; 
			for (Solution sol : this.solutions)
			{
				GraphViz.createDotGraph(((Node)sol.getVariable(0)).getNodeAt(1).todot((long) 1), path+"/model"+(i+1));
				//TreeView btv = new TreeView(((Node)sol.getVariable(0)).getNodeAt(1), 2000, 2000);
				//btv.save(path+"/model"+(i+1)+".png");
				i++ ; 
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static void main(String[] args)
	{
		HashMap<String,Integer> cols = new HashMap<String,Integer>() ; 
		cols.put("dit",0) ; 
		cols.put("noc",0) ; 
		cols.put("cbo",0) ; 
		cols.put("rfc",0) ; 
		cols.put("lcom",0) ; 
		cols.put("ca",0) ; 
		cols.put("ce",0) ; 
		cols.put("npm",0) ; 
		cols.put("lcom3",1) ; 
		cols.put("loc",0) ; 
		cols.put("dam",0) ; 
		cols.put("moa",0) ; 
		cols.put("mfa",1) ; 
		cols.put("cam",1) ; 
		cols.put("ic",0) ; 
		cols.put("cbm",0) ; 
		cols.put("amc",1) ; 
		cols.put("max_cc",0) ; 
		cols.put("avg_cc",1) ; 
        
		String path = "C:/Users/AQ38570/Desktop/data_folds" ; 
		String result_path = "C:/Users/AQ38570/Desktop/results" ; 
		File folder = new File(path);
		File[] listOfFiles = folder.listFiles();
		Boolean success ; 
		String[] metrics = {"G"} ; 
		String Folder_name ; 
		for (int i = 0; i < listOfFiles.length; i++)
		{
			if (listOfFiles[i].getName().contains("train")) 
			{
				CsvDataFrame train = new CsvDataFrame(",");
				CsvDataFrame test = new CsvDataFrame(",");
				try {
					train.read_data_from_file(path+"/"+listOfFiles[i].getName());
					test.read_data_from_file(path+"/"+listOfFiles[i].getName().replace("train", "test"));
					Folder_name = result_path+"/"+listOfFiles[i].getName().replace("train_", "").replace(".csv",""); 
					System.out.println(Folder_name) ; 
					success = (new File(Folder_name).mkdirs());
					if (success == true  || (success == false ) && (new File(Folder_name)).list().length == 0 )
					{	
					    GP_exp gp_exp = new GP_exp(train,test,cols,"bug","GA",metrics,4000,8000000,0.35,0.65) ;
					    gp_exp.save_data(Folder_name); 
					}
					else
						System.out.println(Folder_name + " done :)");
				} catch (FileNotFoundException e) {
					
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				
			}
		}
		/*String Project_name = "train_fold_0_zuzel.csv" ; 
		CsvDataFrame train = new CsvDataFrame(",");
		CsvDataFrame test = new CsvDataFrame(",");
		try {
			train.read_data_from_file(path+"/"+Project_name);
			for (String col : train.get_columns_names())
			{
			    System.out.println(train.get_column_limits(col)) ;
			}	
			test.read_data_from_file(path+"/"+Project_name.replace("train", "test"));
			GP_exp gp_exp = new GP_exp(train,test,cols,"bug","GA",metrics,4000,8000000,0.35,0.65) ;
			Folder_name = result_path+"/"+Project_name.replace("train_", "").replace(".csv",""); 
		    gp_exp.save_data(Folder_name); 
		    NondominatedPopulation sols =  gp_exp.getSolutions() ;
		    for (Solution sol :sols )
		    {
		    	System.out.println(((Node)sol.getVariable(0)).toString());
		    	GraphViz.createDotGraph(((Node)sol.getVariable(0)).getNodeAt(1).todot(1), "C:/Users/AQ38570/Desktop/DotGraph");
		    	System.out.println(((Node)sol.getVariable(0)).getNodeAt(1).todot(1));
		    }
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/

	}
	

}
