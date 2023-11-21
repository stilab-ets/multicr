package core;

import java.util.ArrayList;
import java.util.Collections;
import org.moeaframework.core.Solution;

public class Extended_population {

	ArrayList<Extended_solution> population ; 
	
	public Extended_population()
	{
		this.population = new ArrayList<Extended_solution>() ; 
	}
	public Extended_population(ArrayList<Extended_solution> sols)
	{
		this.population = sols ; 
	}
	
	public void add_solution(Extended_solution e)
	{
		this.population.add(e) ; 
	}
	
	public void reverse_sort()
	{
		Collections.sort(this.population, Collections.reverseOrder());
	}
	
	public void sort()
	{
		Collections.sort(this.population);
	}
	
	public ArrayList<Solution> get_first(int n)
	{
		ArrayList<Solution> result = new ArrayList<Solution>() ;
		for (int i = 0 ; i < Math.min(n, this.population.size()) ; i++)
		{
			result.add(this.population.get(i).getSolution()) ; 
		}
		return result ;
	}
	
	public ArrayList<Extended_solution> getPopulation()
	{
		return this.population ; 
	}
	
	public  ArrayList<Solution> select_best_elements_portion(String criteria,Double portion)
	{
		int size = (int)Math.ceil(portion*population.size()) ;
		Extended_solution.setSortingCriteria(criteria);
		this.reverse_sort();
		ArrayList<Solution> best_solution = this.get_first(size) ; 
		return best_solution ; 
	}
	
	public ArrayList<Solution> filter_based_voting(Double[] thresholds)
	{
		ArrayList<Solution> pop = new ArrayList<Solution>() ;
		for (Extended_solution sol : this.population)
		{
			Boolean selected = true ;
			double[] objectives = sol.getSolution().getObjectives() ; 
			for (int i = 0 ; i < objectives.length ; i++ )
			{
				
				double ival = objectives[i] ; 
				if (thresholds[i] < 0 )
				{
					ival = -1.0*ival ; 
				}
				if (ival < thresholds[i] )
				{
					selected = false ; 
					break ; 
				}
			}
			if (selected) 
			{
				pop.add(sol.getSolution()) ; 
			}
		}
		return pop ; 
	}
}
