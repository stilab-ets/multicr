package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

abstract public class DataFrame {

	protected HashMap<String,HashMap<String,Object>> columns ; 
	protected ArrayList<HashMap<String,Double>> data ; 
	
	public  DataFrame() 
	{
		this.columns = new HashMap<String,HashMap<String,Object>>() ; 
		this.data = new ArrayList<HashMap<String,Double>> ()  ; 
	}
	
	public int count()
	{
		return data.size() ; 
	}
	
	public Set<String> get_columns_names()
	{
		
		return columns.keySet();
	}
	
	public HashMap<String, HashMap<String, Object>> getColumns()
	{
		return this.columns ; 
	}
	
	public HashMap<String,Double> get_row(int index) 
	{
		return this.data.get(index) ; 
	}
	
    public HashMap<String, Object> get_column_limits(String name)
    {
		return this.columns.get(name);
    	
    }
    
    public ArrayList<HashMap<String, Double>> getData()
    {
    	return this.data; 
    }
    public Object[] get_column_data(String name)
    {
    	Object[] col = new Object[this.count()] ; 
    	for (int i = 0 ; i < col.length ; i++)
    	{
    		col[i] = this.get_row(i).get(name) ; 
    	}
		return col;
    }
}
