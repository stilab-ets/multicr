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
		
		/*cols.put("hour",2) ; 
		cols.put("SubmitDay",2) ; 

		cols.put("MessageLength",0) ; 
		cols.put("relatedNum",0) ; 
		cols.put("JavaFileNum",0) ; 
		cols.put("ReviewerNum",0) ; 
		cols.put("ADDING_ATTRIBUTE_MODIFIABILITY",0) ; 
		cols.put("ADDITIONAL_CLASS",0) ; 
		cols.put("ADDITIONAL_FUNCTIONALITY",0) ; 
		cols.put("ADDITIONAL_OBJECT_STATE",0) ; 
		cols.put("ALTERNATIVE_PART_DELETE",0) ; 
		cols.put("ALTERNATIVE_PART_INSERT",0) ; 
		cols.put("ATTRIBUTE_RENAMING",0) ; 
		cols.put("ATTRIBUTE_TYPE_CHANGE",0) ; 
		cols.put("COMMENT_DELETE",0) ; 
		cols.put("COMMENT_INSERT",0) ; 
		cols.put("COMMENT_MOVE",0) ; 
		cols.put("COMMENT_UPDATE",0) ; 
		cols.put("CONDITION_EXPRESSION_CHANGE",0) ; 
		cols.put("DECREASING_ACCESSIBILITY_CHANGE",0) ; 
		cols.put("DOC_DELETE",0) ;
		cols.put("DOC_INSERT",0) ;
		cols.put("DOC_UPDATE",0) ;
		cols.put("INCREASING_ACCESSIBILITY_CHANGE",0) ;
		
		cols.put("METHOD_RENAMING",0) ;
		cols.put("PARAMETER_DELETE",0) ;
		cols.put("PARAMETER_INSERT",0) ;
		cols.put("PARAMETER_ORDERING_CHANGE",0) ;
		cols.put("PARAMETER_RENAMING",0 ) ;
		cols.put("PARAMETER_TYPE_CHANGE",0 ) ;
		cols.put("PARENT_CLASS_CHANGE",0) ;
		cols.put("PARENT_CLASS_DELETE",0) ;
		cols.put("PARENT_CLASS_INSERT",0) ;
		cols.put("PARENT_INTERFACE_INSERT",0) ;
		cols.put("REMOVED_CLASS",0) ;
		cols.put("REMOVED_FUNCTIONALITY",0) ;
		cols.put("REMOVED_OBJECT_STATE",0) ;
		cols.put("REMOVING_ATTRIBUTE_MODIFIABILITY",0 ) ;
		cols.put("REMOVING_CLASS_DERIVABILITY",0 ) ;  
		cols.put("RETURN_TYPE_CHANGE",0 ) ;
		cols.put("RETURN_TYPE_DELETE",0) ;
		cols.put("STATEMENT_DELETE",0) ;  
		cols.put("STATEMENT_INSERT",0) ;  
		cols.put("STATEMENT_ORDERING_CHANGE",0) ;
		cols.put("STATEMENT_PARENT_CHANGE",0) ;  
		cols.put("STATEMENT_UPDATE",0) ;
		cols.put("ChangeRatio",1) ;
		cols.put("degree_centrality",1) ;
		cols.put("closeness_centrality",1) ; 
		cols.put("betweenness_centrality",1) ;
		cols.put("degree",0) ;
		cols.put("in_degree_centrality",1); 
		cols.put("out_degree_centrality",1) ;
		cols.put("review_degree_centrality_review_mean",1) ;
		cols.put("review_degree_centrality_review_sum",1) ;
		cols.put("review_closeness_centrality_review_mean",1) ;
		cols.put("review_closeness_centrality_review_sum",1) ;
		cols.put("review_degree_review_mean",1) ;
		cols.put("review_degree_review_sum",0) ;
		cols.put("review_in_degree_centrality_review_mean",1) ;
		cols.put("review_in_degree_centrality_review_sum",1 ) ;
		cols.put("review_out_degree_centrality_review_mean",1 ) ;
		cols.put("review_out_degree_centrality_review_sum",1) ;
		cols.put("ownerpassratio",1) ;
		cols.put("reviewerpassratio_mean",1) ;
		cols.put("reviewerpassratio_sum",1) ;
		cols.put("reviewerpassratio_max",1) ;
		cols.put("reviewerpassratio_min",1) ;*/
		
		/*cols.put("FOUT_avg",1) ;
		cols.put("FOUT_sum",0) ;
		cols.put("FOUT_max",0) ;
		cols.put("MLOC_avg",1) ;
		cols.put("MLOC_sum",0) ;
		cols.put("MLOC_max",0) ;
		cols.put("NBD_avg",0) ;
		cols.put("NBD_sum",0) ;
		cols.put("NBD_max",1 ) ;
		cols.put("NOF_avg",1 ) ;
		cols.put("NOF_sum",0) ;
		cols.put("NOF_max",0) ;
		cols.put("NOI",0) ;
		cols.put("NOM_avg",1) ;
		cols.put("NOM_sum",0) ;
		cols.put("NOM_max",0) ;
		cols.put("NOT",0) ;
		cols.put("NSF_avg",1 ) ;
		cols.put("NSF_sum",0 ) ;  
		cols.put("NSF_max",0 ) ;
		cols.put("NSM_avg",1) ;
		cols.put("NSM_sum",0) ;  
		cols.put("NSM_max",0) ;  
		cols.put("PAR_avg",1) ;
		cols.put("PAR_sum",0) ;  
		cols.put("PAR_max",0) ;
		cols.put("VG_avg",1) ;
		cols.put("VG_sum",0) ;
		cols.put("pre",0) ; 
		cols.put("VG_max",0) ;
		cols.put("TLOC",0) ;
		cols.put("ACD",0) ;*/
		
		/*cols.put("numberOfVersionsUntil:",0) ;
		cols.put("numberOfFixesUntil:",0) ;
		cols.put("numberOfRefactoringsUntil:",0) ;
		cols.put("numberOfAuthorsUntil:",0) ;
		cols.put("linesAddedUntil:",0) ;
		cols.put("maxLinesAddedUntil:",0) ;
		cols.put("avgLinesAddedUntil:",1) ;
		cols.put("linesRemovedUntil:",0) ;
		cols.put("maxLinesRemovedUntil:",0 ) ;
		cols.put("avgLinesRemovedUntil:",1 ) ;
		cols.put("codeChurnUntil:",0) ;
		cols.put("maxCodeChurnUntil:",0) ;
		cols.put("avgCodeChurnUntil:",1) ;
		cols.put("ageWithRespectTo:",1) ;
		cols.put("weightedAgeWithRespectTo:",1) ;*/
		
		/*cols.put("additions",0) ;
		cols.put("additions_avg",0) ;
		cols.put("additions_max",0) ;
		cols.put("avg_play_size",1) ;
		cols.put("avg_task_size",1) ;
		cols.put("change_set_avg",0) ;
		cols.put("change_set_max",0) ;
		cols.put("code_churn_avg",0) ;
		cols.put("code_churn_count",0) ;
		cols.put("code_churn_max",0) ;
		cols.put("commits_count",0) ;
		cols.put("contributors_count",0) ;
		cols.put("deletions",0) ;
		cols.put("deletions_avg",0) ;
		cols.put("deletions_max",0) ;
		cols.put("highest_contributor_experience",1) ;
		cols.put("hunks_median",1) ;
		cols.put("lines_blank",0) ;
		cols.put("lines_code",0) ;
		cols.put("lines_comment",0) ;
		cols.put("minor_contributors_count",0) ;
		cols.put("num_authorized_key",0) ;
		cols.put("num_block_error_handling",0) ;
		cols.put("num_blocks",0) ;
		cols.put("num_commands",0) ;
		cols.put("num_conditions",0) ;
		cols.put("num_decisions",0) ;
		cols.put("num_deprecated_keywords",0) ;
		cols.put("num_deprecated_modules",0) ;
		cols.put("num_distinct_modules",0) ;
		cols.put("num_external_modules",0) ;
		cols.put("num_fact_modules",0) ;
		cols.put("num_file_exists",0) ;
		cols.put("num_file_mode",0) ;
		cols.put("num_file_modules",0) ;
		cols.put("num_filters",0) ;
		cols.put("num_ignore_errors",0) ;
		cols.put("num_import_playbook",0) ;
		cols.put("num_import_role",0) ;
		cols.put("num_import_tasks",0) ;
		cols.put("num_include",0) ;
		cols.put("num_include_role",0) ;
		cols.put("num_include_tasks",0) ;
		cols.put("num_include_vars",0) ;
		cols.put("num_keys",0) ;
		cols.put("num_lookups",0) ;
		cols.put("num_loops",0) ;
		cols.put("num_math_operations",0) ;
		cols.put("num_names_with_vars",0) ;
		cols.put("num_parameters",0) ;
		cols.put("num_paths",0) ;
		cols.put("num_plays",0) ;
		cols.put("num_prompts",0) ;
		cols.put("num_regex",0) ;
		cols.put("num_roles",0) ;
		cols.put("num_suspicious_comments",0) ;
		cols.put("num_tasks",0) ;
		cols.put("num_tokens",0) ;
		cols.put("num_unique_names",0) ;
		cols.put("num_uri",0) ;
		cols.put("num_vars",0) ;
		cols.put("text_entropy",1) ;

		cols.put("delta_avg_play_size",1) ;
		cols.put("delta_avg_task_size",1) ;
		cols.put("delta_lines_blank",0) ;
		cols.put("delta_lines_code",0) ;
		cols.put("delta_lines_comment",0) ;
		cols.put("delta_num_authorized_key",0) ;
		cols.put("delta_num_block_error_handling",0) ;
		cols.put("delta_num_blocks",0) ;
		cols.put("delta_num_commands",0) ;
		cols.put("delta_num_conditions",0) ;
		cols.put("delta_num_decisions",0) ;
		cols.put("delta_num_deprecated_keywords",0) ;
		cols.put("delta_num_deprecated_modules",0) ;
		cols.put("delta_num_distinct_modules",0) ;
		cols.put("delta_num_external_modules",0) ;
		cols.put("delta_num_fact_modules",0) ;
		cols.put("delta_num_file_exists",0) ;
		cols.put("delta_num_file_mode",0) ;
		cols.put("delta_num_file_modules",0) ;
		cols.put("delta_num_filters",0) ;
		cols.put("delta_num_ignore_errors",0) ;
		cols.put("delta_num_import_playbook",0) ;
		cols.put("delta_num_import_role",0) ;
		cols.put("delta_num_import_tasks",0) ;
		cols.put("delta_num_include",0) ;
		cols.put("delta_num_include_role",0) ;
		cols.put("delta_num_include_tasks",0) ;
		cols.put("delta_num_include_vars",0) ;
		cols.put("delta_num_keys",0) ;
		cols.put("delta_num_lookups",0) ;
		cols.put("delta_num_loops",0) ;
		cols.put("delta_num_math_operations",0) ;
		cols.put("delta_num_names_with_vars",0) ;
		cols.put("delta_num_parameters",0) ;
		cols.put("delta_num_paths",0) ;
		cols.put("delta_num_plays",0) ;
		cols.put("delta_num_prompts",0) ;
		cols.put("delta_num_regex",0) ;
		cols.put("delta_num_roles",0) ;
		cols.put("delta_num_suspicious_comments",0) ;
		cols.put("delta_num_tasks",0) ;
		cols.put("delta_num_tokens",0) ;
		cols.put("delta_num_unique_names",0) ;
		cols.put("delta_num_uri",0) ;
		cols.put("delta_num_vars",0) ;
		cols.put("delta_text_entropy",1) ;*/
		
		/*cols.put("NoD", 0 ) ; 
		cols.put("NoCD", 0 ) ; 
		cols.put("NCD", 0 ) ; 
		cols.put("SDC", 1 ) ; 
		cols.put("PCD", 0 ) ; 
		cols.put("NAD",1 ) ; 
		cols.put("NSD",0 ) ;
		cols.put("PSD",1 ) ;
		cols.put("NPR",0 ) ; 
		cols.put("SAPR",1 ) ; 
		cols.put("ANAP",1 ) ;
		cols.put("NIS",1) ;
		cols.put("SDAI",1 ) ;
		cols.put("ANAI",1 ) ;
		cols.put("GDC",1 ) ;
		cols.put("SDD",1 ) ;
		cols.put("GBC",1 ) ;
		cols.put("GCC",1 ) ;
		cols.put("ND",1 ) ;
		cols.put("NC",1 ) ;
		cols.put("ACC",1 ) ;
		cols.put("SCC",1 ) ;
		cols.put("SDoC",1 ) ;
		cols.put("TZ",0 ) ;
		cols.put("ACZ",1 ) ;
		cols.put("SCZ",1 ) ;
		cols.put("ADZ",1 ) ;
		cols.put("SDZ",1 ) ;
		cols.put("NR",0 ) ; 
		cols.put("NAD",1 ) ;
		cols.put("PCR",1) ;
		cols.put("SCR",1 ) ;
		cols.put("FN",1 ) ;
		cols.put("ADPR",0 ) ;
		cols.put("ADI",0 ) ;
		cols.put("BFN",1) ; 
		cols.put("TFC",1) ;
		cols.put("ANCPR",1 ) ;
		cols.put("SCPR",1) ; 
		cols.put("NCI",1 ) ;
		cols.put("ANCI",1) ;
		cols.put("SDCI",1) ;
		cols.put("RTCPR",1) ;
		cols.put("RTCI",1) ; 
		cols.put("RPCPR",1) ;
		cols.put("RINC",1) ;
		cols.put("RNSPRC",1) ;
		cols.put("ACCL",1) ; */
		 
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
		
		//cols.put("text_prob_ngram",1); 
		// multiple revisions 
		//cols.put("duration",1); 
		//cols.put("number_of_message",0); 
		//cols.put("number_of_revision",0); 
		//cols.put("avg_delay_between_revision",1); 
		//cols.put("weighted_approval_score",1); */
		//jira features 
		/*cols.put("CountDeclMethodPrivate",0); 
		cols.put("AvgLineCode",0); 
		cols.put("CountLine",0); 
		cols.put("MaxCyclomatic",0); 
		cols.put("CountDeclMethodDefault",0); 
		cols.put("AvgEssential",0); 
		cols.put("CountDeclClassVariable",0); 
		cols.put("SumCyclomaticStrict",0); 
		cols.put("AvgCyclomatic",0); 
		cols.put("AvgLine",0); 
		cols.put("CountDeclClassMethod",0); 
		cols.put("AvgLineComment",0); 
		cols.put("AvgCyclomaticModified",0); 
		cols.put("CountDeclFunction",0); 
		cols.put("CountLineComment",0); 
		cols.put("CountDeclClass",0); 
		cols.put("CountDeclMethod",0); 
		cols.put("SumCyclomaticModified",0); 
		cols.put("CountLineCodeDecl",0); 
		cols.put("CountDeclMethodProtected",0); 
		cols.put("CountDeclInstanceVariable",0); 
		cols.put("MaxCyclomaticStrict",0); 
		cols.put("CountDeclMethodPublic",0); 
		cols.put("CountLineCodeExe",0); 
		cols.put("SumCyclomatic",0); 
		cols.put("SumEssential",0); 
		cols.put("CountStmtDecl",0); 
		cols.put("CountLineCode",0); 
		cols.put("CountStmtExe",0); 
		cols.put("RatioCommentToCode",0); 
		cols.put("CountLineBlank",0); 
		cols.put("CountStmt",0); 
		cols.put("MaxCyclomaticModified",0); 
		cols.put("CountSemicolon",0); 
		cols.put("AvgLineBlank",0); 
		cols.put("CountDeclInstanceMethod",0); 
		cols.put("AvgCyclomaticStrict",0); 
		cols.put("PercentLackOfCohesion",0); 
		cols.put("MaxInheritanceTree",0); 
		cols.put("CountClassDerived",0); 
		cols.put("CountClassCoupled",0); 
		cols.put("CountClassBase",0); 
		cols.put("CountInput_Max",0); 
		cols.put("CountInput_Mean",1); 
		cols.put("CountInput_Min",0); 
		cols.put("CountOutput_Max",0); 
		cols.put("CountOutput_Mean",1); 
		cols.put("CountOutput_Min",0); 
		cols.put("CountPath_Max",0); 
		cols.put("CountPath_Mean",1); 
		cols.put("CountPath_Min",0); 
		cols.put("MaxNesting_Max",0); 
		cols.put("MaxNesting_Mean",1); 
		cols.put("MaxNesting_Min",0); 
		cols.put("COMM",0); 
		cols.put("ADEV",0); 
		cols.put("DDEV",0); 
		cols.put("Added_lines",0); 
		cols.put("Del_lines",0); 
		cols.put("OWN_LINE",1); 
		cols.put("OWN_COMMIT",0); 
		cols.put("MINOR_COMMIT",0); 
		cols.put("MINOR_LINE",0); 
		cols.put("MAJOR_COMMIT",0); 
		cols.put("MAJOR_LINE",0);*/
		
		//cols.put("FFT_prediction",0); 
		/*cols.put("wmc",0) ; 
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
		cols.put("dam",1) ; 
		cols.put("moa",0) ; 
		cols.put("mfa",1) ; 
		cols.put("cam",1) ; 
		cols.put("ic",0) ; 
		cols.put("cbm",0) ; 
		cols.put("amc",1) ; 
		cols.put("max_cc",0) ; 
		cols.put("avg_cc",1) ; */
		
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
					
					Run_MOGP nsga2_run = new Run_MOGP(train,cols,TARGET,"nsga2",multi_metrics,400,2400,0.8,0.15,new ArrayList<Solution>()) ; 
					Run_MOGP spea2_run = new Run_MOGP(train,cols,TARGET,"spea2",multi_metrics,500,1200,0.8,0.15,new ArrayList<Solution>()) ; 
					Run_MOGP ibea_run = new Run_MOGP(train,cols,TARGET,"ibea",multi_metrics,400,1000,0.9,0.9,new ArrayList<Solution>()) ; 
					Run_MOGP simple_nsga3_run = new Run_MOGP(train,cols,TARGET,"nsga3",multi_metrics,400,2400,0.9,0.9
							,new ArrayList<Solution>()) ; 
					
					Run_MOGP extended_nsga3_run = new Run_MOGP(train,cols,TARGET,"nsga3",ibea_metrics,200,1000,0.7,0.15,new ArrayList<Solution>()) ; 
					Run_MOGP extended_ibea_run = new Run_MOGP(train,cols,TARGET,"ibea",ibea_metrics,1024,200,0.7,0.1,new ArrayList<Solution>()) ; 
					
					Test_Single_Model test_mono_gp = new Test_Single_Model(val, test,mono_gp_run,best_point) ; 
					Test_Single_Model test_tpr_gp = new Test_Single_Model(val, test,tpr_gp_run,best_point) ; 
					Test_Single_Model test_tnr_gp = new Test_Single_Model(val, test,tnr_gp_run,best_point) ; 
	
					Test_Single_Model test_rs = new Test_Single_Model(val, test,rs_run,best_point) ; 
					
					Test_Single_Model test_nsga2 = new Test_Single_Model(val, test,nsga2_run,best_point) ; 
					Test_Single_Model test_spea2 = new Test_Single_Model(val, test,spea2_run,best_point) ; 
					Test_Single_Model test_ibea = new Test_Single_Model(val, test,ibea_run,best_point) ; 
					Test_Single_Model test_simple_nsga3 = new Test_Single_Model(val, test,simple_nsga3_run,best_point) ;  
					
					Test_Single_Model test_extended_nsga3 = new Test_Single_Model(val, test,extended_nsga3_run,best_point) ;
					Test_Single_Model test_extended_ibea = new Test_Single_Model(val, test,extended_ibea_run,best_point) ;
					
					
					Run_exp mono_gp_exp = new Run_exp(Folder_name+"/mono_gp",mono_gp_run,test_mono_gp) ; 
					
					Run_exp tpr_gp_exp = new Run_exp(Folder_name+"/tpr_gp",tpr_gp_run,test_tpr_gp) ; 
					Run_exp tnr_gp_exp = new Run_exp(Folder_name+"/tnr_gp",tnr_gp_run,test_tnr_gp) ; 
	
					Run_exp rs_exp = new Run_exp(Folder_name+"/rs",rs_run,test_rs) ; 
					
					Run_exp nsga2_exp = new Run_exp(Folder_name+"/nsga2",nsga2_run,test_nsga2) ; 
					Run_exp spea2_exp = new Run_exp(Folder_name+"/spea2",spea2_run,test_spea2) ; 
					Run_exp simple_nsga3_exp = new Run_exp(Folder_name+"/nsga3",simple_nsga3_run,test_simple_nsga3) ; 
					Run_exp ibea_exp = new Run_exp(Folder_name+"/ibea",ibea_run,test_ibea) ; 
	
					
					Run_exp extended_nsga3_exp = new Run_exp(Folder_name+"/extended_nsga3",extended_nsga3_run,test_extended_nsga3) ; 
					Run_exp extended_ibea_exp = new Run_exp(Folder_name+"/extended_ibea",extended_ibea_run,test_extended_ibea) ; 
					
					//mono_gp_exp.start() ; 
					//tpr_gp_exp.start();
					//tnr_gp_exp.start();
					//rs_exp.start() ; 
					//nsga2_exp.start();
					//spea2_exp.start() ; 
					//simple_nsga3_exp.start();
					ibea_exp.start();
					//extended_nsga3_exp.start();
				    //extended_ibea_exp.start();
					try {
						//mono_gp_exp.join();
						//tpr_gp_exp.join();
						//tnr_gp_exp.join();
						//rs_exp.join();
						//nsga2_exp.join();
						//spea2_exp.join() ; 
						//simple_nsga3_exp.join();
						ibea_exp.join();
						//extended_nsga3_exp.join();
						//extended_ibea_exp.join();
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}	
				}
			}
		}	
	}
}
