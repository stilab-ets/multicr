/* Copyright 2009-2016 David Hadka
 *
 * This file is part of the MOEA Framework.
 *
 * The MOEA Framework is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * The MOEA Framework is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with the MOEA Framework.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.moeaframework.util.tree;

/**
 * The node for calculating the logical AND of two boolean values.  The inputs
 * and outputs to this node are shown below:
 * 
 * <table border="1" cellpadding="3" cellspacing="0">
 *   <tr class="TableHeadingColor">
 *     <th width="25%" align="left">Name</th>
 *     <th width="25%" align="left">Type</th>
 *     <th width="50%" align="left">Description</th>
 *   </tr>
 *   <tr>
 *     <td>Argument 1</td>
 *     <td>Boolean</td>
 *     <td>The first boolean value</td>
 *   </tr>
 *   <tr>
 *     <td>Argument 2</td>
 *     <td>Boolean</td>
 *     <td>The second boolean value</td>
 *   </tr>
 *   <tr>
 *     <td>Return Value</td>
 *     <td>Number</td>
 *     <td>The logical AND of the two boolean values</td>
 *   </tr>
 * </table>
 */
public class Nor extends Node {
	
	/**
	 * Constructs a new node for calculating the logical AND of two boolean
	 * values.
	 */
	public Nor() {
		super(Boolean.class, Boolean.class, Boolean.class);
	}

	@Override
	public Nor copyNode() {
		return new Nor();
	}

	@Override
	public Boolean evaluate(Environment environment) {
		Boolean arg_one = (Boolean)getArgument(0).evaluate(environment) ;
		Boolean arg_two = (Boolean)getArgument(1).evaluate(environment);
		return !(arg_one || arg_two) ;
	}

}
