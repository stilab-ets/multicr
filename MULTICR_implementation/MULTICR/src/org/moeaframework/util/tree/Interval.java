package org.moeaframework.util.tree;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

public class Interval extends Node {
	String name ; 
	Double min ; 
	Double max ; 
	Double [] bounds ; 
	int type ; 
	public Interval(String name, Double min, Double max,int type,Double [] bounds) {
		super(Boolean.class) ;
		this.name = new String(name) ; 
		this.min = min ; 
		this.max = max ; 
		this.type = type ; 
		this.bounds = bounds ; 
	
	}

	@Override
	public Interval copyNode() {
		return new Interval(name,min,max,type,bounds) ; 
	}

	@Override
	public Boolean evaluate(Environment environment) {
		if (this.bounds == null)
		{
			this.bounds = new Double[2] ; 
			if (type == 1 )
			{
				//System.out.println("here!") ;
				// Continuous variable
				this.bounds[0] = this.min + Math.random()*(this.max - this.min) ; 
				this.bounds[1] = this.min + Math.random()*(this.max - this.min) ; 
			}
			else 
			{
				Random randomno = new Random();
				this.bounds[0] = (double) randomno.nextInt(this.max.intValue()+1) + this.min ; 
				randomno =  new Random(); 
				this.bounds[1] = (double) randomno.nextInt(this.max.intValue()+1) + this.min ; 
			}
			Arrays.sort(bounds);
		}
		Double value = environment.get(Double.class, this.name) ; 
		//System.out.println(this.toString()) ; 
		if (value >= bounds[0] && value <= bounds[1])
		{
			return true ;
		}
		return false ; 
	}
	@Override
    public String toString()
    {
		DecimalFormat df = new DecimalFormat("#.##");
		df.setRoundingMode(RoundingMode.CEILING);
		return (df.format(this.bounds[0]) +  " <= " +name + " <= " + df.format(this.bounds[1]));
    	
    }
	@Override
	public String getName()
	{
		return this.toString() ; 
	}
}
