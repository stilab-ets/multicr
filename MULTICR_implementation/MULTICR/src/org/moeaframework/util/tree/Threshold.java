package org.moeaframework.util.tree;

import java.util.Random ; 

public abstract class Threshold extends Node {
	
	public final String name ;
	final Double min_val ;
	final Double max_val ;
	Double val ;
	protected int type ; 
	
	Threshold(String name,double min,double max,double val,int type)
	{
		super(Boolean.class) ;
		this.name = name ;
		this.min_val = min ;
		this.max_val = max ;
		this.val = val ;
		this.type = type ; 
		
	}
	protected void set_threshold()
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
	
}