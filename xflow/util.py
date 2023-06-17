import random
import numpy as np

def run (graph, diffusion, seeds, method, eval, epoch, budget, output):
    print("Running " + eval.upper() + " :")

    for graph_fn in graph:
        try:
            print(graph_fn.__name__)
            g, config = graph_fn()
            print(g)

            if eval == 'im' or eval == 'ibm':

                seeds = random.sample(list(g.nodes()), 10)

                for method_fn in method:
                    try:
                        print(method_fn.__name__)
                        baselines = ['eigen', 'degree', 'pi', 'sigma', 'Netshield', 'IMRank']
                        if method_fn.__name__ in baselines:
                            m = method_fn(g, config, budget=10)
                        baselines = ['RIS']
                        if method_fn.__name__ in baselines:
                            m = method_fn(g, config, budget=10)
                        baselines = ['greedy', 'celf', 'celfpp']
                        if method_fn.__name__ in baselines:
                            for diffusion_fn in diffusion:
                                try:
                                    print(diffusion_fn.__name__)
                                    if eval == 'im':
                                        m = method_fn(g, config, budget, rounds=epoch, model=diffusion_fn.__name__, beta=0.1)
                                    if eval == 'ibm':
                                        m = method_fn(g, config, budget, seeds, rounds=epoch, model=diffusion_fn.__name__, beta=0.1)
                                except Exception as e:
                                    print(f"Error when calling {diffusion_fn.__name__}: {str(e)}")
                        baselines = ['netsleuth']
                        if method_fn.__name__ in baselines:
                            m = method_fn(I, g, hypotheses_per_step=1)
                            
                        
                    except Exception as e:
                        print(f"Error when calling {method_fn.__name__}: {str(e)}")    

            if eval == 'sl':
                # TODO: seed should be changable
                seed = 10
                random.seed(seed)
                np.random.seed(seed)

                contagion = method.static_network_contagion.StaticNetworkContagion(
                    G=g,
                    model="si",
                    infection_rate=0.1,
                    # recovery_rate=0.005, # for SIS/SIR models
                    number_infected = 3,
                    seed=seed
                )

                contagion.forward(steps = 16)
                
                step = 15

                # This obtains the indices of all vertices in the infected category at the 15th step of the simulation.
                I = contagion.get_infected_subgraph(step=step)

        except Exception as e:
            print(f"Error when calling {graph_fn.__name__}: {str(e)}")