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
import org.moeaframework.core.PRNG;
/**
 * The node for defining a constant value.  The inputs and outputs to this node
 * are shown below:
 * 
 * <table border="1" cellpadding="3" cellspacing="0">
 *   <tr class="TableHeadingColor">
 *     <th width="25%" align="left">Name</th>
 *     <th width="25%" align="left">Type</th>
 *     <th width="50%" align="left">Description</th>
 *   </tr>
 *   <tr>
 *     <td>Return Value</td>
 *     <td>Number</td>
 *     <td>The constant value</td>
 *   </tr>
 * </table>
 */
public class EphermeralConstant extends Node {
	
	/**
	 * The value.
	 */
	private Number value;
	private Number min ; 
	private Number max ; 
	
	
	public EphermeralConstant(Number min, Number max) {
		super(Number.class);
		
		this.value = null;
		this.min = min ; 
		this.max = max ; 
	}
	
	@Override
	public EphermeralConstant copyNode() {
		return new EphermeralConstant(this.min, this.max);
	}
	
	@Override
	public Object evaluate(Environment environment) {
		if (this.value == null)
		{
			this.value = PRNG.nextDouble((Double)this.min, (Double)this.max) ; 
		}
		return (Double)value;
	}
	
	@Override
	public String toString() {
		return String.valueOf(value);
	}
	public String getName()
	{
		return this.toString() ; 
	}
}
