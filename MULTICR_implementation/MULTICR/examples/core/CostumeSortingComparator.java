package core;

import java.util.Comparator;

import org.moeaframework.core.Solution;

public class CostumeSortingComparator implements Comparator<Solution> {

	String criteria ; 
	public CostumeSortingComparator(String criteria)
	{
		this.criteria = criteria ;
	}
	@Override
	public int compare(Solution o1, Solution o2) {
		
		int critirea_compare = Double.valueOf((String) o1.get_external_Attribute(criteria)).compareTo(Double.valueOf((String) o2.get_external_Attribute(criteria))) ; 
		return critirea_compare;
	}

}
