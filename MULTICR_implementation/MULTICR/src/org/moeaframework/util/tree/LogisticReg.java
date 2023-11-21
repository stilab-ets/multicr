package org.moeaframework.util.tree;

import java.math.RoundingMode;
import java.text.DecimalFormat;

public class LogisticReg extends Node{
	Double threshold = 0.5 ; 
	public LogisticReg(Class<?>[] argumentTypes) {
		// TODO Auto-generated constructor stub
		super(Boolean.class, argumentTypes) ; 
	}

	@Override
	public Node copyNode() {
		// TODO Auto-generated method stub
		return new LogisticReg(this.getArgumentTypes());
	}

	

	@Override
	public Object evaluate(Environment environment) {
		// TODO Auto-generated method stub
		Double res = 0.0 ; 
		for (int iargument = 0 ; iargument < this.getNumberOfArguments();  iargument++) {
			Double final_val = 0.0 ; 
			if (this.getArgument(iargument).getReturnType() == Boolean.class) {
				//System.out.println(this.getArgument(iargument).getReturnType()) ; 
				//System.out.println("I m here!!") ; 
				Boolean val = (Boolean)this.getArgument(iargument).evaluate(environment); 
				if (val) {
					final_val = 1.0 ; 
				}
				else {
					final_val = 0.0 ; 
				}
			}
			else
				final_val = (Double)this.getArgument(iargument).evaluate(environment); 
			res += res +final_val ; 
		}
		if (1/(1 + Math.exp(-res)) > this.threshold) {
			//System.out.println("here!!!") ; 
			return (Boolean)true ; 
			
		}
		else  {
			//System.out.println(" and here!!!") ; 
			return (Boolean)false ; 
		}
	}
	@Override
    public String toString()
    {
		
		return "logit >= " + this.threshold;
    	
    }
	@Override
	public String getName()
	{
		return this.toString() ; 
	}

}
