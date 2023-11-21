package MOEA_GP;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.Map.Entry;

import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.Program;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.util.tree.* ; 
import core.ConfusionMatrix;
import core.CsvDataFrame;
import core.DataFrame;

public class Train_MOGP extends AbstractProblem {
	
	private DataFrame train_data ; 
    Object[] true_labels  ; 
    private Rules rules;
    private Set<String> features ; 
    private String[] Objectives ; 
    private ArrayList<Node> must_contain; 
    
	public Train_MOGP(DataFrame data,Object[] true_labels, HashMap<String,Integer> feature_columns,String[] Objectives)
	{
		
		super(1, Objectives.length, 0);
		
		this.Objectives = Objectives ; 
        this.features = feature_columns.keySet() ; 
        this.train_data = data;
		this.rules = new Rules();
		this.rules.add(new And());
		this.rules.add(new Or());
		
		this.must_contain = new ArrayList<Node>() ; 
		
		
	    rules.setReturnType(Boolean.class);
		rules.setMaxInitializationDepth(10);
		rules.setFunctionCrossoverProbability(0.8);
		rules.setMaxVariationDepth(10);
		
		this.true_labels = true_labels ; 
		for (Entry<String, Integer> entry : feature_columns.entrySet())
		{
			HashMap<String, Object> feature_limits = this.train_data.get_column_limits(entry.getKey()) ;
			
		
			if (entry.getValue() < 2) {
				this.rules.add(new Greater_than_threshold(entry.getKey(),(Double)feature_limits.get("min"),(Double) feature_limits.get("max"),-1.0,entry.getValue()));
				this.rules.add(new Lesser_than_threshold(entry.getKey(),(Double)feature_limits.get("min"),(Double) feature_limits.get("max"),-1.0,entry.getValue()));
			}
			if (entry.getValue() > 2) {
				this.rules.add(new EqualThreshold(entry.getKey(),(Double)feature_limits.get("min"),(Double) feature_limits.get("max"),-1.0, 0)) ; 
			}
			
			
		}
		
	}
	@Override
	public void evaluate(Solution solution)
	{
		Boolean [] outputs =Gp_try.compute_predictions(solution,this.train_data.getData(),features)  ; 
		ConfusionMatrix matrix = new ConfusionMatrix(this.true_labels,(Object[])outputs) ;
		
		for (int i = 0 ; i < this.numberOfObjectives ; i++)
		{
			if (this.Objectives[i] == "G")
			{
				solution.setObjective(i,-1*matrix.G_measure());
			}
			if (this.Objectives[i] == "accuracy")
			{
				solution.setObjective(i,-1*matrix.accuracy());
			}
			if (this.Objectives[i] == "f1")
			{
				solution.setObjective(i,-1*matrix.F1_measure());
			}
			if (this.Objectives[i] == "tnr")
			{
				solution.setObjective(i,-1*matrix.specificity());
			}
			if (this.Objectives[i] == "tpr")
			{
				solution.setObjective(i,-1*matrix.sensitivity());
			}
			if (this.Objectives[i] == "maxheight")
			{
				solution.setObjective(i,((Node)solution.getVariable(0)).getMaximumHeight());
			}
			if (this.Objectives[i] == "nodenumber")
			{
				//solution.setObjective(i,((Node)solution.getVariable(0)).getNumberOfNodes());
				int functions_number = ((Node)solution.getVariable(0)).getNumberOfFunctions() ;
				int terminal_number = ((Node)solution.getVariable(0)).getNumberOfTerminals() ;
				solution.setObjective(i,(functions_number + terminal_number)*1.0/32); 
				int n_nodes = ((Node)solution.getVariable(0)).getNumberOfNodes() ; 
				/*if (n_nodes <= 5) {
					solution.setConstraint(0, 0);
				}
				else {
					solution.setConstraint(0, n_nodes - 9);
				}*/
				if (n_nodes  >= 7 ) {
					solution.setConstraint(0, 0.0);
				}
				else {
						solution.setConstraint(0, 7 - n_nodes);
				}
			}
			if (this.Objectives[i] == "-fpr")
			{
				solution.setObjective(i,1*matrix.fpr());
			}
			if (this.Objectives[i] == "f1")
			{
				solution.setObjective(i,-1*matrix.F1_measure());
			}
			if (this.Objectives[i] == "G")
			{
				solution.setObjective(i,-1*matrix.G_measure());
			}
			if (this.Objectives[i] == "-fnr")
			{
				solution.setObjective(i,1*matrix.fnr());
			}
			if (this.Objectives[i] == "tpr-fpr")
			{
				solution.setObjective(i,-1*(matrix.specificity() + (1-matrix.fpr())));
			}
			if (this.Objectives[i] == "tnr-fnr")
			{
				solution.setObjective(i,-1*(matrix.sensitivity() + (1-matrix.fnr())));
			}
			if (this.Objectives[i] == "precision")
			{
				solution.setObjective(i,-1*(matrix.precision()));
			}
			if (this.Objectives[i] == "mean_tpr_tnr")
			{
				solution.setObjective(i,-1*0.5*(matrix.sensitivity() + matrix.specificity()));
			}
			if (this.Objectives[i] == "MCC")
			{
				solution.setObjective(i,-1*(matrix.MCC()));
			}
			if (this.Objectives[i] == "-balance")
			{
				solution.setObjective(i,1*(matrix.balance()));
			}
			if (this.Objectives[i] == "MCC_times_f1")
			{
				solution.setObjective(i,-1*(matrix.MCC()*matrix.F1_measure()));
			}
			if (this.Objectives[i] == "-MisCost")
			{
				double c = 1.0 ; 
				double alpha = 5.0 ; 
				double fp = matrix.getFp() ; 
				double fn = matrix.getFn() ;
				double cost = (fn + alpha*fp)*1.0/this.train_data.count(); 
				solution.setObjective(i,cost);
			}
			
			
		}
		/*Double tpr = matrix.specificity()  ; 
		Double tnr = matrix.sensitivity() ; 
	    System.out.println("tpr:"+tpr+" tnr:"+tnr) ; 
		System.out.println("acc:"+matrix.accuracy()) ; 
		System.out.println("f1:"+matrix.F1_measure()) ; 
		if (tpr > 0.9 && tnr > 0.9)
			System.out.println("WoW!!") ;  
		solution.setObjective(0,-1*tpr);
		solution.setObjective(1,-1*tnr);*/
	}
	public int countNumMustExistNode(Solution solution) {
		ArrayList<Node> terminal_list = ((Node)solution.getVariable(0)).getTerminals() ; 
		int num_node_exist = 0 ;
		Greater_than_threshold check_exist_greater = null ; 
		Lesser_than_threshold check_exist_lesser  = null ;
		Equal_threshold check_exist_equal  = null ;
		for (Node must_exist: this.must_contain) {
			if (must_exist instanceof Greater_than_threshold) {
				check_exist_greater = (Greater_than_threshold) must_exist;
			}
			if (must_exist instanceof Lesser_than_threshold) {
				check_exist_lesser = (Lesser_than_threshold) must_exist;
			}
			
			if (must_exist instanceof Equal_threshold) {
				check_exist_equal = (Equal_threshold) must_exist;
			}
			
			for (Node terminal: terminal_list) {
				if (check_exist_greater != null) {
					if (terminal instanceof Greater_than_threshold) {
						if (check_exist_greater.is_equal((Greater_than_threshold)terminal)) {
							num_node_exist += 1 ; 
							break; 
						}
					}
				}
				if (check_exist_lesser != null) {
					if (terminal instanceof Lesser_than_threshold) {
						if (check_exist_lesser.is_equal((Lesser_than_threshold)terminal)) {
							num_node_exist += 1 ; 
							break; 
						}
					}
				}
				if (check_exist_equal != null) {
					if (terminal instanceof Equal_threshold) {
						if (check_exist_equal.is_equal((Equal_threshold)terminal)) {
							num_node_exist += 1 ; 
							break; 
						}
					}
				}
			}
		}
		return num_node_exist ; 
	}
	@Override
	public Solution newSolution() { 
		Solution solution = new Solution(1, this.Objectives.length,this.numberOfConstraints); 
		solution.setVariable(0, new Program(rules));
		return solution;
	}
}
