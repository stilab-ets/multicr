package MOEA_GP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;

import org.moeaframework.core.Solution;

import core.*;

public class Run_exp extends Thread {

	Thread t ; 
	String path;
	Run_MOGP run ;
	Test_Single_Model test ; 
	public Run_exp(String path,Run_MOGP run,Test_Single_Model test) 
	{
		this.run = run ; 
		this.test = test ; 
		this.path = path ; 
	}
	@Override 
	public void run()
	{
		if (this.run.train_done == false)
		this.run.run();
		
		this.test.save(this.path);
	}
	
	public static void main(String[] args)
	{
		HashMap<String,Integer> cols = new HashMap<String,Integer>() ; 	
		
	    // code review metrics 
		
		cols.put("total_change_num",0); 
		cols.put("num_of_reviewers",0); 
		cols.put("description_length",0); 
		cols.put("is_documentation",3); 
		cols.put("is_bug_fixing",3); 
		cols.put("is_feature",3); 
		cols.put("project_changes_per_week",1); 
		cols.put("project_merge_ratio",1); 
		cols.put("changes_per_author",1); 
		cols.put("num_of_bot_reviewers",0); 
		cols.put("lines_added",0); 
		cols.put("lines_deleted",0); 
		cols.put("files_added",0); 
		cols.put("files_deleted",0); 
		cols.put("files_modified",0); 
		cols.put("num_of_directory",0); 
		cols.put("modify_entropy",1); 
		cols.put("subsystem_num",0); 
		
		
		String TARGET = "status" ; 
		int nruns = 10 ;
		String datapath = "C:/Users/Motaz/Desktop/work/code_review_delay_prediction/early_abondon_prediction/GP_data/Eclipse"  ; //"C:/Users/AQ38570/Desktop/test_data" ; 
		String result_path = "C:/Users/Motaz/Desktop/work/code_review_delay_prediction/early_abondon_prediction/GP_res_reduced_rules/Eclipse" ; 
	 	File folder = new File(datapath);
		File[] listOfFiles = folder.listFiles();
		
		String[] multi_metrics = {"tpr","tnr"} ; 
		String[] ibea_metrics = {"tpr","tnr","-fpr","-fnr"} ;
		String[] single_metric = {"MCC"} ; 
		String[] tpr_metric = {"tpr"} ; 
		String[] tnr_metric = {"tnr"} ; 
		double[] best_point = {1.0,1.0} ; 
		for (int i = 0; i < listOfFiles.length; i++)
		{
			if (listOfFiles[i].getName().contains("train")) 
			{	
				CsvDataFrame train = new CsvDataFrame(",");
				CsvDataFrame  test= new CsvDataFrame(",") ; 
				CsvDataFrame val= new CsvDataFrame(",") ; 
				
			
					System.out.println("reading data") ; 
					System.out.println(datapath+"/"+listOfFiles[i].getName()) ; 
					try {
						train.read_data_from_file(datapath+"/"+listOfFiles[i].getName());
						//val.read_data_from_file(datapath+"/"+listOfFiles[i].getName().replace("train", "val"));
						test.read_data_from_file(datapath+"/"+listOfFiles[i].getName().replace("train", "test"));
					}
					catch (FileNotFoundException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
						continue ;
					}
					
				val = train ; 
				System.out.println(train.count());
				//System.out.println(val.count());
				System.out.println(test.count());
				for (int run = 0 ; run < nruns ; run++) {
					String Folder_name = result_path+"/"+listOfFiles[i].getName().replace("train_", "").replace(".csv","") + "/run_"+run;
					if (new File(Folder_name).exists())
					{
						System.out.println(Folder_name + " found !") ; 
						continue ; 
					}
					Run_MOGP mono_gp_run = new Run_MOGP(train,cols,TARGET,"ga",single_metric,500,200,0.9,0.1,new ArrayList<Solution>()) ; 
					
					Run_MOGP tpr_gp_run = new Run_MOGP(train,cols,TARGET,"ga",tpr_metric,500,200,0.9,0.1,new ArrayList<Solution>()) ; 
					Run_MOGP tnr_gp_run = new Run_MOGP(train,cols,TARGET,"ga",tnr_metric,500,200,0.9,0.1,new ArrayList<Solution>()) ; 
	
					Run_MOGP rs_run = new Run_MOGP(train,cols,TARGET,"Random",multi_metrics,500,200,0.75,0.15,new ArrayList<Solution>()) ; 
					
					Run_MOGP nsga2_run = new Run_MOGP(train,cols,TARGET,"nsga2",multi_metrics,500,200,0.9,0.1,new ArrayList<Solution>()) ; 
					Run_MOGP spea2_run = new Run_MOGP(train,cols,TARGET,"spea2",multi_metrics,500,1200,0.9,0.1,new ArrayList<Solution>()) ; 
					Run_MOGP ibea_run = new Run_MOGP(train,cols,TARGET,"ibea",multi_metrics,400,1000,0.9,0.9,new ArrayList<Solution>()) ; 
					Run_MOGP simple_nsga3_run = new Run_MOGP(train,cols,TARGET,"nsga3",multi_metrics,400,2400,0.9,0.9
							,new ArrayList<Solution>()) ; 
					
					
					Test_Single_Model test_mono_gp = new Test_Single_Model(val, test,mono_gp_run,best_point) ; 
					Test_Single_Model test_tpr_gp = new Test_Single_Model(val, test,tpr_gp_run,best_point) ; 
					Test_Single_Model test_tnr_gp = new Test_Single_Model(val, test,tnr_gp_run,best_point) ; 
	
					Test_Single_Model test_rs = new Test_Single_Model(val, test,rs_run,best_point) ; 
					
					Test_Single_Model test_nsga2 = new Test_Single_Model(val, test,nsga2_run,best_point) ; 
					Test_Single_Model test_spea2 = new Test_Single_Model(val, test,spea2_run,best_point) ; 
					Test_Single_Model test_ibea = new Test_Single_Model(val, test,ibea_run,best_point) ; 
					Test_Single_Model test_simple_nsga3 = new Test_Single_Model(val, test,simple_nsga3_run,best_point) ;  
					
					
					
					Run_exp mono_gp_exp = new Run_exp(Folder_name+"/mono_gp",mono_gp_run,test_mono_gp) ; 
					
					Run_exp tpr_gp_exp = new Run_exp(Folder_name+"/tpr_gp",tpr_gp_run,test_tpr_gp) ; 
					Run_exp tnr_gp_exp = new Run_exp(Folder_name+"/tnr_gp",tnr_gp_run,test_tnr_gp) ; 
	
					Run_exp rs_exp = new Run_exp(Folder_name+"/rs",rs_run,test_rs) ; 
					
					Run_exp nsga2_exp = new Run_exp(Folder_name+"/nsga2",nsga2_run,test_nsga2) ; 
					Run_exp spea2_exp = new Run_exp(Folder_name+"/spea2",spea2_run,test_spea2) ; 
					Run_exp simple_nsga3_exp = new Run_exp(Folder_name+"/nsga3",simple_nsga3_run,test_simple_nsga3) ; 
					Run_exp ibea_exp = new Run_exp(Folder_name+"/ibea",ibea_run,test_ibea) ; 
	
					
					
					//mono_gp_exp.start() ; 
					//tpr_gp_exp.start();
					//tnr_gp_exp.start();
					//rs_exp.start() ; 
					//nsga2_exp.start();
					//spea2_exp.start() ; 
					//simple_nsga3_exp.start();
					ibea_exp.start();
					try {
						//mono_gp_exp.join();
						//tpr_gp_exp.join();
						//tnr_gp_exp.join();
						//rs_exp.join();
						//nsga2_exp.join();
						//spea2_exp.join() ; 
						//simple_nsga3_exp.join();
						ibea_exp.join();
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}	
				}
			}
		}	
	}
}
