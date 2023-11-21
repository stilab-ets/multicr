package org.moeaframework.util.tree;

public class OR3 extends Node {

	public OR3() {
		super(Boolean.class, Boolean.class, Boolean.class, Boolean.class);
	}

	@Override
	public OR3 copyNode() {
		return new OR3();
	}

	@Override
	public Object evaluate(Environment environment) {
		Boolean arg_one = (Boolean)getArgument(0).evaluate(environment) ;
		Boolean arg_two = (Boolean)getArgument(1).evaluate(environment);
		Boolean arg_three = (Boolean)getArgument(2).evaluate(environment);
		return arg_one || arg_two || arg_three;
	}
}
