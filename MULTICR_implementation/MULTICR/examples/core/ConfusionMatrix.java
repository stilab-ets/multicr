package core;

import java.util.HashMap;
import java.util.Map;

public class ConfusionMatrix {
	
	private int tp ; 
	private int tn ; 
	private int fp ; 
	private int fn ;
	
	public ConfusionMatrix(Object[] True_values,Object[] predected_values){
		this.tp = 0 ; 
		this.tn = 0 ; 
		this.fp = 0 ; 
		this.fn = 0 ; 
		
		for (int i = 0 ; i < predected_values.length ; i++)
		{
			Boolean actual = false ; 
			if ((Double)True_values[i] > 0.0)
			{
				actual = true ; 
			}
			//System.out.println("predected= "+predected_values[i]) ; 
			//System.out.println("actual = "+actual) ; 
			if ((Boolean)predected_values[i] == true)
			{
				if (predected_values[i] == actual)
				{
					this.tp++ ; 
				}
				else
				{
					this.fp++ ; 
				}
			}
			if ((Boolean)predected_values[i] == false)
			{
				if (predected_values[i] == actual)
				{
					this.tn++ ; 
				}
				else
				{
					this.fn++ ; 
				}
			}
		}
		/*System.out.println("tp:"+this.tp);
		System.out.println("tn:"+this.tn);
		System.out.println("fp:"+this.fp);
		System.out.println("fn:"+this.fn);*/
	}
	
	public Double precision()
	{
		if (this.tp == 0)
			return 0.0 ; 
		return (this.tp+0.0)/(this.tp+this.fp) ; 
	}
	public Double sensitivity()
	{
		if (this.tp == 0)
			return 0.0 ; 
		return (this.tp+0.0)/(this.tp+this.fn) ; 
	}
	public Double specificity()
	{
		if (this.tn == 0)
			return 0.0 ; 
		return (this.tn+0.0)/(this.tn+this.fp) ; 
	}
	public Double accuracy()
	{
		return (this.tp+this.tn+0.0)/(this.tp+this.tn+this.fp+this.fn) ; 
	}
    public Double F1_measure()
    {
    	if (this.precision() == 0 || this.sensitivity() == 0)
    		return 0.0 ;
    	return 2.0*this.precision()*this.sensitivity()/(this.precision() + this.sensitivity() ) ; 
    }
    public Double G_measure()
    {   
    	return Math.sqrt(this.sensitivity()*this.specificity()) ; 
    }
    public Double balance()
	{
		return Math.sqrt((Math.sqrt(Math.pow(fpr(), 2) + Math.pow(1 - sensitivity(),2))) / 2);
	}
    public Double MCC()
    {
    	Double dem = Math.sqrt(this.tn+this.fn)*Math.sqrt(this.tn+this.fp)*Math.sqrt(this.tp+this.fn)*Math.sqrt(this.tp+this.fp) ;
    	if (Double.compare(dem, 0.0)==0) return -1.0;
    	Double c = (this.tp*this.tn - this.fp*this.fn)/dem ;
    	return c;
    }
    public Double fpr()
    {
    	return this.fp*1.0/(this.fp + this.tn) ;
    }
    public Double fnr()
    {
    	return this.fn*1.0/(this.fn + this.tp) ;
    }
    
	public int getTn() {
		return tn;
	}

	public void setTn(int tn) {
		this.tn = tn;
	}

	public int getTp() {
		return tp;
	}

	public void setTp(int tp) {
		this.tp = tp;
	}

	public int getFp() {
		return fp;
	}

	public void setFp(int fp) {
		this.fp = fp;
	}

	public int getFn() {
		return fn;
	}

	public void setFn(int fn) {
		this.fn = fn;
	}
	
	public Map<String, Object> get_statistics()
	{
		Map<String, Object> statistics = new HashMap<String, Object>() ; 
		statistics.put("tp", String.valueOf(this.tp*1.0)) ; 
		statistics.put("tn", String.valueOf(this.tn*1.0)) ; 
		statistics.put("fp", String.valueOf(this.fp*1.0)) ; 
		statistics.put("fn", String.valueOf(this.fn*1.0)) ; 
		statistics.put("tpr", String.valueOf(this.sensitivity())) ; 
		statistics.put("tnr", String.valueOf(this.specificity())) ; 
		statistics.put("F1", String.valueOf(this.F1_measure())) ; 
		statistics.put("MCC", String.valueOf(this.MCC())) ; 
		statistics.put("G", String.valueOf(this.G_measure())) ; 
		statistics.put("accuracy", String.valueOf(this.accuracy())) ;
		statistics.put("fpr", String.valueOf(this.fpr())) ; 
		statistics.put("fnr", String.valueOf(this.fnr())) ; 
		statistics.put("precision", String.valueOf(this.precision())) ; 
		statistics.put("balance", String.valueOf(this.balance())) ; 
		
		return statistics ; 
		
		
	}

}
