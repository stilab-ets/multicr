package org.moeaframework.util.tree;

public class AND3 extends Node {
	public AND3() {
		super(Boolean.class, Boolean.class, Boolean.class, Boolean.class);
	}

	@Override
	public AND3 copyNode() {
		return new AND3();
	}

	@Override
	public Object evaluate(Environment environment) {
		Boolean arg_one = (Boolean)getArgument(0).evaluate(environment) ;
		Boolean arg_two = (Boolean)getArgument(1).evaluate(environment);
		Boolean arg_three = (Boolean)getArgument(2).evaluate(environment);
		return arg_one && arg_two && arg_three;
	}

}
