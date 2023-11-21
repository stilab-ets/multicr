package MOEA_GP;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Properties;

import org.apache.commons.math3.util.CombinatoricsUtils;
import org.moeaframework.Executor;
import org.moeaframework.algorithm.IBEA;
import org.moeaframework.algorithm.NSGAII;
import org.moeaframework.algorithm.ReferencePointNondominatedSortingPopulation;
import org.moeaframework.algorithm.SPEA2;
import org.moeaframework.algorithm.single.AggregateObjectiveComparator;
import org.moeaframework.algorithm.single.GeneticAlgorithm;
import org.moeaframework.algorithm.single.LinearDominanceComparator;
import org.moeaframework.algorithm.single.MinMaxDominanceComparator;
import org.moeaframework.core.Algorithm;
import org.moeaframework.core.FrameworkException;
import org.moeaframework.core.Initialization;
import org.moeaframework.core.NondominatedPopulation;
import org.moeaframework.core.NondominatedSortingPopulation;
import org.moeaframework.core.PRNG;
import org.moeaframework.core.Population;
import org.moeaframework.core.Problem;
import org.moeaframework.core.Selection;
import org.moeaframework.core.Solution;
import org.moeaframework.core.Variation;
import org.moeaframework.core.comparator.AggregateConstraintComparator;
import org.moeaframework.core.comparator.ChainedComparator;
import org.moeaframework.core.comparator.CrowdingComparator;
import org.moeaframework.core.comparator.DominanceComparator;
import org.moeaframework.core.comparator.ParetoDominanceComparator;
import org.moeaframework.core.fitness.AdditiveEpsilonIndicatorFitnessEvaluator;
import org.moeaframework.core.fitness.HypervolumeFitnessEvaluator;
import org.moeaframework.core.fitness.IndicatorFitnessEvaluator;
import org.moeaframework.core.operator.InjectedInitialization;
import org.moeaframework.core.operator.RandomInitialization;
import org.moeaframework.core.operator.TournamentSelection;
import org.moeaframework.core.spi.OperatorFactory;
import org.moeaframework.core.spi.ProviderLookupException;
import org.moeaframework.core.spi.ProviderNotFoundException;
import org.moeaframework.util.TypedProperties;

import core.ConfusionMatrix;
import core.DataFrame;
import core.Extended_population;
import core.Extended_solution;

public class Run_MOGP {
	
	
	public NondominatedPopulation solutions ; 
	public HashMap<String,String> run_configs ; 
	public DataFrame train_data ; 
	public HashMap<String,Integer> features ;
	public String topredict ; 
	public String [] objectives ; 
	public Object[] true_train_labels ; 
	public ArrayList<Solution> initial_population ;
	public Boolean train_done ;
	public Extended_population final_solution ; 
	public Long fit_time ; 
	
	public Run_MOGP(DataFrame train_data, HashMap<String,Integer> features,String topredict,String algo,String[] objectives,int pop_size,int maxgeneration,Double cross_prob,Double mut_prob,ArrayList<Solution> intitial_pop) {
		
		
		//saving run parameters 
		this.train_data = train_data ; 
		this.run_configs = new HashMap<String,String>() ; 
		this.features = features ;
		this.objectives = objectives ;
		this.topredict = topredict ; 
		this.true_train_labels = train_data.get_column_data(topredict) ; 
		this.initial_population =  intitial_pop ;
		this.final_solution = new Extended_population() ; 
		this.train_done = false  ;
		
		
		this.run_configs.put("algorithm", algo); 
		this.run_configs.put("populationsize",  String.valueOf(pop_size)); 
		this.run_configs.put("mutation_rate",  String.valueOf(mut_prob)); 
		this.run_configs.put("crossover_rate",  String.valueOf(cross_prob)); 
		this.run_configs.put("maxgeneration",  String.valueOf(maxgeneration));
		this.run_configs.put("algorithm", algo); 
		
	 
		
		
	}

	private Algorithm IBEA_With_Intitial_Pop(TypedProperties properties, Problem problem,List<Solution> initial_pop) {
		if (problem.getNumberOfConstraints() > 0) {
			throw new ProviderNotFoundException("IBEA", 
					new ProviderLookupException("constraints not supported"));
		}
		
		int populationSize = (int)properties.getDouble("populationSize", 100);
		String indicator = properties.getString("indicator", "hypervolume");
		IndicatorFitnessEvaluator fitnessEvaluator = null;

		Initialization initialization = new InjectedInitialization(problem,
				populationSize,initial_pop);

		Variation variation = OperatorFactory.getInstance().getVariation(null, 
				properties, problem);
		
		if ("hypervolume".equals(indicator)) {
			fitnessEvaluator = new HypervolumeFitnessEvaluator(problem);
		} else if ("epsilon".equals(indicator)) {
			fitnessEvaluator = new AdditiveEpsilonIndicatorFitnessEvaluator(
					problem);
		} else {
			throw new IllegalArgumentException("invalid indicator: " +
					indicator);
		}

		return new IBEA(problem, null, initialization, variation,
				fitnessEvaluator);
	}
	
	private Algorithm SPEA2_With_Intitial_Pop(TypedProperties properties, Problem problem,List<Solution> initial_pop) {
		int populationSize = (int)properties.getDouble("populationSize", 100);
		int offspringSize = (int)properties.getDouble("offspringSize", 100);
		int k = (int)properties.getDouble("k", 1);
		
		Initialization initialization = new InjectedInitialization(problem,
				populationSize,initial_pop);

		Variation variation = OperatorFactory.getInstance().getVariation(null, 
				properties, problem);

		return new SPEA2(problem, initialization, variation, offspringSize, k);
	}
	
	private Algorithm NSGAIII_With_Intitial_Pop(TypedProperties properties, Problem problem,List<Solution> initial_pop) {
		int divisionsOuter = 4;
		int divisionsInner = 0;
		
		if (properties.contains("divisionsOuter") && properties.contains("divisionsInner")) {
			divisionsOuter = (int)properties.getDouble("divisionsOuter", 4);
			divisionsInner = (int)properties.getDouble("divisionsInner", 0);
		} else if (properties.contains("divisions")){
			divisionsOuter = (int)properties.getDouble("divisions", 4);
		} else if (problem.getNumberOfObjectives() == 1) {
			divisionsOuter = 100;
		} else if (problem.getNumberOfObjectives() == 2) {
			divisionsOuter = 99;
		} else if (problem.getNumberOfObjectives() == 3) {
			divisionsOuter = 12;
		} else if (problem.getNumberOfObjectives() == 4) {
			divisionsOuter = 8;
		} else if (problem.getNumberOfObjectives() == 5) {
			divisionsOuter = 6;
		} else if (problem.getNumberOfObjectives() == 6) {
			divisionsOuter = 4;
			divisionsInner = 1;
		} else if (problem.getNumberOfObjectives() == 7) {
			divisionsOuter = 3;
			divisionsInner = 2;
		} else if (problem.getNumberOfObjectives() == 8) {
			divisionsOuter = 3;
			divisionsInner = 2;
		} else if (problem.getNumberOfObjectives() == 9) {
			divisionsOuter = 3;
			divisionsInner = 2;
		} else if (problem.getNumberOfObjectives() == 10) {
			divisionsOuter = 3;
			divisionsInner = 2;
		} else {
			divisionsOuter = 2;
			divisionsInner = 1;
		}
		
		int populationSize;
		
		if (properties.contains("populationSize")) {
			populationSize = (int)properties.getDouble("populationSize", 100);
		} else {
			// compute number of reference points
			populationSize = (int)(CombinatoricsUtils.binomialCoefficient(problem.getNumberOfObjectives() + divisionsOuter - 1, divisionsOuter) +
					(divisionsInner == 0 ? 0 : CombinatoricsUtils.binomialCoefficient(problem.getNumberOfObjectives() + divisionsInner - 1, divisionsInner)));

			// round up to a multiple of 4
			populationSize = (int)Math.ceil(populationSize / 4d) * 4;
		}
		
		Initialization initialization = new InjectedInitialization(problem,
				populationSize,initial_pop) ;
		
		ReferencePointNondominatedSortingPopulation population = new ReferencePointNondominatedSortingPopulation(
				problem.getNumberOfObjectives(), divisionsOuter, divisionsInner);

		Selection selection = null;
		
		if (problem.getNumberOfConstraints() == 0) {
			selection = new Selection() {
	
				@Override
				public Solution[] select(int arity, Population population) {
					Solution[] result = new Solution[arity];
					
					for (int i = 0; i < arity; i++) {
						result[i] = population.get(PRNG.nextInt(population.size()));
					}
					
					return result;
				}
				
			};
		} else {
			selection = new TournamentSelection(2, new ChainedComparator(
					new AggregateConstraintComparator(),
					new DominanceComparator() {

						@Override
						public int compare(Solution solution1, Solution solution2) {
							return PRNG.nextBoolean() ? -1 : 1;
						}
						
					}));
		}
		
		// disable swapping variables in SBX operator to remain consistent with
		// Deb's implementation (thanks to Haitham Seada for identifying this
		// discrepancy)
		if (!properties.contains("sbx.swap")) {
			properties.setBoolean("sbx.swap", false);
		}
		
		if (!properties.contains("sbx.distributionIndex")) {
			properties.setDouble("sbx.distributionIndex", 30.0);
		}
		
		if (!properties.contains("pm.distributionIndex")) {
			properties.setDouble("pm.distributionIndex", 20.0);
		}

		Variation variation = OperatorFactory.getInstance().getVariation(null, 
				properties, problem);

		return new NSGAII(problem, population, null, selection, variation,
				initialization);
	}
	
	private Algorithm GeneticAlgorithm_With_Intitial_Pop(TypedProperties properties, Problem problem,List<Solution> initial_pop) {
		int populationSize = (int)properties.getDouble("populationSize", 100);
		double[] weights = properties.getDoubleArray("weights", new double[] { 1.0 });
		String method = properties.getString("method", "linear");
		
		AggregateObjectiveComparator comparator = null;
		
		if (method.equalsIgnoreCase("linear")) {
			comparator = new LinearDominanceComparator(weights);
		} else if (method.equalsIgnoreCase("min-max")) {
			comparator = new MinMaxDominanceComparator(weights);
		} else {
			throw new FrameworkException("unrecognized weighting method: " + method);
		}

		Initialization initialization = new  InjectedInitialization(problem,
				populationSize,initial_pop) ;
		
		Selection selection = new TournamentSelection(2, comparator);

		Variation variation = OperatorFactory.getInstance().getVariation(null, properties, problem);

		return new GeneticAlgorithm(problem, comparator, initialization, selection, variation);
	}
	private Algorithm newNSGAII(TypedProperties properties, Problem problem) {
		int populationSize = (int)properties.getDouble("populationSize", 100);

		Initialization initialization = new RandomInitialization(problem,
				populationSize);

		NondominatedSortingPopulation population = 
				new NondominatedSortingPopulation();

		TournamentSelection selection = null;
		
		if (properties.getBoolean("withReplacement", true)) {
			selection = new TournamentSelection(2, new ChainedComparator(
					new ParetoDominanceComparator(),
					new CrowdingComparator()));
		}

		Variation variation = OperatorFactory.getInstance().getVariation(null, 
				properties, problem);

		return new NSGAII(problem, population, null, selection, variation,
				initialization);
	}
	
	public Algorithm get_algorithm(String algo,Properties properties, Problem problem,List<Solution> initial_pop )
	{
		Algorithm algorithm = null;
		if (algo == "nsga3") 
		{
			algorithm =  NSGAIII_With_Intitial_Pop(new TypedProperties(properties),problem,initial_pop) ;
		}
		if (algo == "nsga2")
		{
			algorithm = newNSGAII(new TypedProperties(properties),problem) ; 
		}
		if (algo == "ibea") 
		{
			algorithm =  IBEA_With_Intitial_Pop(new TypedProperties(properties),problem,initial_pop) ;
		}
		
		if (algo == "spea2") 
		{
			algorithm =  SPEA2_With_Intitial_Pop(new TypedProperties(properties),problem,initial_pop) ;
		}
		
		if (algo == "ga") 
		{
			algorithm =  GeneticAlgorithm_With_Intitial_Pop(new TypedProperties(properties),problem,initial_pop) ;
		}
		return algorithm ;
	}
	
	public void run()
	{
		if (this.train_done)
		{
			return ;
		}
		Long start,end;
		start = System.currentTimeMillis();
		Problem problem =  new Train_MOGP(train_data,true_train_labels,features,objectives) ;
		if (initial_population.isEmpty())
		{
			if (this.run_configs.get("algorithm") != "Random")
			{
				System.out.println("here") ; 
				this.solutions = new Executor()
				      .withProblemClass(Train_MOGP.class, train_data,true_train_labels,features,objectives)
				      .distributeOn(10)
				      .withAlgorithm(this.run_configs.get("algorithm"))
				      .withProblem(problem)
				      .distributeOnAllCores()
				      .withProperty("populationSize", this.run_configs.get("populationsize"))
				      .withProperty("operator", "bx")
				      .withProperty("operator", "ptm")
				      .withProperty("bx.rate",Double.parseDouble(this.run_configs.get("crossover_rate")))
				      .withProperty("ptm.rate", Double.parseDouble(this.run_configs.get("mutation_rate")))
				      .withMaxEvaluations(Integer.parseInt(this.run_configs.get("maxgeneration"))*Integer.parseInt(this.run_configs.get("populationsize")))
				      .run();  
			      //.withMaxTime(30*60*1000)

			}
			else 
			{
				System.out.println("poor random search") ; 
				//this.objectives = new String[]{"none"};
				this.solutions = new Executor()
					      .withProblemClass(Train_MOGP.class, train_data,true_train_labels,features,this.objectives)
					      .withAlgorithm("Random")
					      //.withProblem(problem)
					      .withProperty("populationSize",Integer.parseInt(this.run_configs.get("maxgeneration"))*Integer.parseInt(this.run_configs.get("populationsize")))
					      .withMaxEvaluations(Integer.parseInt(this.run_configs.get("maxgeneration"))*Integer.parseInt(this.run_configs.get("populationsize")) + 1)
					      .withEpsilon(0.01)
					      .distributeOnAllCores()
					      .run();  
			}
		}
		
		else {
			
			//running algorithm with initial population 
			Properties properties = new Properties();
			properties.setProperty("populationSize",this.run_configs.get("populationsize"));
			properties.setProperty("operator","bx");
			properties.setProperty("operator","ptm");
			properties.setProperty("bx.rate",this.run_configs.get("crossover_rate")) ; 
			properties.setProperty("ptm.rate",this.run_configs.get("mutation_rate")) ; 
		
			Algorithm algorithm = null;
			algorithm = this.get_algorithm( this.run_configs.get("algorithm"), properties,problem,this.initial_population) ;
			int maxGenerations = Integer.parseInt(this.run_configs.get("maxgeneration"));
				
			for (int igen = 0 ; igen< maxGenerations ; igen++)
			{
				System.out.println("generation : " + (igen+1));
				algorithm.step(); 
				this.solutions = algorithm.getResult() ;
			}
			
		}
		this.train_done = true ;
		end = System.currentTimeMillis();
		this.fit_time = end - start ; 
		
		// creating extended solution 
		for (Solution sol : this.solutions)
		{
			ConfusionMatrix matrix = new ConfusionMatrix(this.true_train_labels,Gp_try.compute_predictions(sol, this.train_data.getData(), this.features.keySet()));
			//System.out.println(matrix.get_statistics().toString()) ; 
			sol.add_external_attributes(matrix.get_statistics());
			//System.out.println(sol.getAttributes()) ; 
			//this.final_solution.add_solution(new Extended_solution(sol,new ConfusionMatrix(this.true_train_labels,Gp_try.compute_predictions(sol, this.train_data.getData(), this.features.keySet()))));
		}
	}
	/* public void start () {
	      System.out.println("Starting " +  this.run_configs.get("algorithm") );
	      if (t == null) {
	         t = new Thread (this, this.run_configs.get("algorithm"));
	         t.start ();
	      }
	 }*/
}