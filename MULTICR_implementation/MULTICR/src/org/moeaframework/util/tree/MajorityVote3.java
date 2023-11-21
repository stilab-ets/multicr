package org.moeaframework.util.tree;

public class MajorityVote3 extends Node {

	public MajorityVote3() {
		super(Boolean.class, Boolean.class, Boolean.class, Boolean.class);
	}
	@Override
	public MajorityVote3 copyNode() {
		// TODO Auto-generated method stub
		return new MajorityVote3();
	}

	@Override
	public Object evaluate(Environment environment) {
		Boolean arg_one = (Boolean)getArgument(0).evaluate(environment) ;
		Boolean arg_two = (Boolean)getArgument(1).evaluate(environment);
		Boolean arg_three = (Boolean)getArgument(2).evaluate(environment);
		return (arg_one && arg_two) || (arg_one && arg_three) || (arg_two && arg_three);
	}

}
