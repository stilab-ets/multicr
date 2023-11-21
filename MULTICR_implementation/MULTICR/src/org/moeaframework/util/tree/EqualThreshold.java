package org.moeaframework.util.tree;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.Random;

public class EqualThreshold extends Threshold {
	public EqualThreshold(String name, Double min, Double max, Double val,int type) {
		super(name, min, max, val,type);
	
	}

	@Override
	public EqualThreshold copyNode() {
		return new EqualThreshold(name,min_val,max_val,val,type) ; 
	}

	@Override
	public Boolean evaluate(Environment environment) {
		if (this.val == -1)
		{
			if (type == 1 )
			{
				//System.out.println("here!") ;
				// Continuous variable
				this.val = this.min_val + Math.random()*(this.max_val - this.min_val) ; 
			}
			else 
			{
				Random randomno = new Random();
				this.val = (double) randomno.nextInt(this.max_val.intValue()+1 - this.min_val.intValue()) + this.min_val ; 
			}
		}
		Double value = environment.get(Double.class, this.name) ; 
		//System.out.println(this.toString()) ; 
		return (value == this.val.doubleValue()) ;
	}
	@Override
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
	public boolean is_equal(EqualThreshold candidate) {
		return (this.name.equalsIgnoreCase(candidate.name)) && (this.val == candidate.val) ; 
	}
}
