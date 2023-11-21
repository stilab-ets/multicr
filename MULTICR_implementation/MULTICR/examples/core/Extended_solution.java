package core;

import org.moeaframework.core.Solution;

public class Extended_solution implements Comparable<Extended_solution> {
	
	public static String sorting_criteria = "G" ;
	
	private Solution solution ;
	private ConfusionMatrix train_confusion_matrix ;
	
	public Extended_solution(Solution sol, ConfusionMatrix matrix)
	{
		this.solution = sol ;
		this.train_confusion_matrix = matrix ; 
	}
	
	public Solution getSolution()
	{
		return this.solution ;
	}
	
	public ConfusionMatrix getMatrix()
	{
		return this.train_confusion_matrix ;
	}
	
	public String getSortingCriteria()
	{
		return sorting_criteria ;
	}
	
	public static void setSortingCriteria(String criteria)
	{
		sorting_criteria = criteria ;
	}
	
	@Override
	public int compareTo(Extended_solution o) {
		Double val = Double.parseDouble((String) o.getMatrix().get_statistics().get(sorting_criteria));

		return Double.compare(Double.parseDouble((String) this.getMatrix().get_statistics().get(sorting_criteria)),val) ; 
	}
	 
}