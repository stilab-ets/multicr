package MOEA_GP;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.moeaframework.core.Solution;
import org.moeaframework.core.variable.Program;
import org.moeaframework.problem.AbstractProblem;
import org.moeaframework.util.tree.Add;
import org.moeaframework.util.tree.And ;
import org.moeaframework.util.tree.Constant;
import org.moeaframework.util.tree.Divide;
import org.moeaframework.util.tree.Environment;
import org.moeaframework.util.tree.Get;
import org.moeaframework.util.tree.GreaterThan;
import org.moeaframework.util.tree.GreaterThanOrEqual;
import org.moeaframework.util.tree.Or ;
import org.moeaframework.util.tree.Rules;

import core.ConfusionMatrix;
import core.CsvDataFrame;
import core.DataFrame;

import org.moeaframework.util.tree.Greater_than_threshold ;
import org.moeaframework.util.tree.LessThan;
import org.moeaframework.util.tree.LessThanOrEqual;
import org.moeaframework.util.tree.Lesser_than_threshold ;
import org.moeaframework.util.tree.Log;
import org.moeaframework.util.tree.Max;
import org.moeaframework.util.tree.Min;
import org.moeaframework.util.tree.Multiply;
import org.moeaframework.util.tree.Node;
import org.moeaframework.util.tree.Not; 

public class Gp_try extends AbstractProblem {
	
	private DataFrame train_data ; 
	public String path ; 
    Object[] true_labels  ; 
    private Rules rules;
    private Set<String> features ; 
    
	public Gp_try(String path,String topredict, HashMap<String,Integer> feature_columns) {
		
		super(1, 1, 0);
        this.features = feature_columns.keySet() ; 
		this.path = path ;
		
		this.rules = new Rules();
		this.rules.add(new And());
		this.rules.add(new Or());
		//this.rules.add(new Add());
		//this.rules.add(new Divide());
		//this.rules.add(new Multiply());
		//this.rules.add(new Max());
		//this.rules.add(new Min());
		//this.rules.add(new Log());
		//this.rules.add(new GreaterThanOrEqual());
		//this.rules.add(new LessThanOrEqual());
		/*this.rules.add(new GreaterThan());
	    this.rules.add(new LessThan());
	    this.rules.add(new Constant(true));
	    this.rules.add(new Constant(false));*/
	    rules.setReturnType(Boolean.class);
		rules.setMaxInitializationDepth(11);
		train_data = new CsvDataFrame(",") ;
		
		try {
			((CsvDataFrame) train_data).read_data_from_file(this.path);
			for (String col : this.train_data.get_columns_names())
			{
				System.out.println(col) ; 
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} 
		true_labels = train_data.get_column_data(topredict) ; 
		for (Entry<String, Integer> entry : feature_columns.entrySet())
		{
			HashMap<String, Object> feature_limits = this.train_data.get_column_limits(entry.getKey()) ; 
			System.out.println(entry.getKey());
			this.rules.add(new Greater_than_threshold(entry.getKey(),(Double)feature_limits.get("min"),(Double) feature_limits.get("max"),Double.NaN,entry.getValue()));
			this.rules.add(new Lesser_than_threshold(entry.getKey(),(Double)feature_limits.get("min"),(Double) feature_limits.get("max"),Double.NaN,entry.getValue()));
			//this.rules.add(new Get(Double.class,entry.getKey()));
			
		}
		
	}

	@Override
	public void evaluate(Solution solution) {
		Boolean [] outputs = Gp_try.compute_predictions(solution,this.train_data.getData(),features)  ; 
		ConfusionMatrix matrix = new ConfusionMatrix(this.true_labels,(Object[])outputs) ; 
		Double G_score = matrix.G_measure()  ; 
		System.out.println(G_score);
		System.out.println("acc:"+matrix.accuracy()) ; 
		System.out.println("f1:"+matrix.F1_measure()) ; 
		System.out.println("G:"+G_score) ; 
		if (G_score > 0.9)
			System.out.println("WoW!!") ;  
		solution.setObjective(0,-1*G_score);
	}

	@Override
	public Solution newSolution() { 
		Solution solution = new Solution(1, 1); 
		solution.setVariable(0, new Program(rules));
		return solution;
	}
	public static Boolean [] compute_predictions(Solution solution,ArrayList<HashMap<String,Double>> data,Set<String> features) 
	{
		Program program = (Program)solution.getVariable(0);
		
		Boolean[] outputs = new Boolean[ data.size()];
		int i = 0 ; 
		for (HashMap<String, Double> input_example :data )
		{
			Environment environment = new Environment(); 
			for (String column : features )
			{
				environment.set(column, (Double)input_example.get(column));
			}
			outputs[i] = (Boolean) program.evaluate(environment);
			i++ ; 
		}
		
		return outputs ; 
		
	}

}
