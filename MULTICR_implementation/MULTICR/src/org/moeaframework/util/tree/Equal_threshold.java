package org.moeaframework.util.tree;

import java.math.RoundingMode;
import java.text.DecimalFormat;

public class Equal_threshold extends Threshold {

	public Equal_threshold(String name, double min, double max, double val, int type) {
		super(name, min, max, val, type);
		// TODO Auto-generated constructor stub
	}

	@Override
	public Node copyNode() {
		// TODO Auto-generated method stub
		return  new Equal_threshold(name,min_val,max_val,val,type) ;
	}

	@Override
	public Object evaluate(Environment environment) {
		Double value = environment.get(Double.class, this.name) ;
		return (value == this.val.doubleValue()) ;
	}
	public String toString()
    {
		DecimalFormat df = new DecimalFormat("#.##");
		df.setRoundingMode(RoundingMode.CEILING);
		return name + " = " + df.format(this.val);
    	
    }
	@Override
	public String getName()
	{
		return this.toString() ; 
	}
	public boolean is_equal(Equal_threshold candidate) {
		return (this.name.equalsIgnoreCase(candidate.name)) && (this.val == candidate.val) ; 
	}

}
