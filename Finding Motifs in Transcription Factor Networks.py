"""
Lab 8: Finding Motifs in Transcription Factor Networks  
Author: RK Azhigulova  
Date: 24 March 2025
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import seaborn as sns

# Set style for more appealing visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

class TLR4_NFkB_Model:
    """
    A mathematical model of the TLR4-MyD88-TAK1-NEMO-p50/p65 signaling pathway.
    This model uses ordinary differential equations to simulate the dynamics of 
    the NFkB signaling pathway upon LPS stimulation.
    """
    
    def __init__(self):
        # Set default parameter values
        self.params = {
            # Activation/deactivation rates
            'k_lps_tlr4': 0.1,      # LPS binding to TLR4
            'k_tlr4_myd88': 0.2,    # TLR4 activation of MyD88
            'k_myd88_tak1': 0.3,    # MyD88 activation of TAK1
            'k_tak1_ikk': 0.2,      # TAK1 activation of IKK complex
            'k_ikk_ikb': 0.5,       # IKK phosphorylation of IkB
            'k_ikb_deg': 0.4,       # IkB degradation rate
            'k_nfkb_nuclear': 0.7,  # NFkB nuclear translocation rate
            'k_nfkb_cytoplasmic': 0.05,  # NFkB export from nucleus
            
            # Protein synthesis rates
            'k_ikb_synthesis': 0.05, # IkB synthesis rate
            'k_a20_synthesis': 0.04, # A20 synthesis rate
            'k_gene_transcription': 0.1, # Gene transcription rate
            
            # Degradation/inactivation rates
            'gamma_tlr4': 0.03,     # TLR4 inactivation rate
            'gamma_myd88': 0.03,    # MyD88 inactivation rate
            'gamma_tak1': 0.05,     # TAK1 inactivation rate
            'gamma_ikk': 0.05,      # IKK inactivation rate
            'gamma_nfkb': 0.01,     # NFkB degradation rate
            'gamma_ikb': 0.1,       # IkB degradation rate
            'gamma_a20': 0.05,      # A20 degradation rate
            'gamma_mrna': 0.03,     # mRNA degradation rate
            
            # Inhibition constants
            'ki_a20_tak1': 0.2,     # A20 inhibition of TAK1
            'ki_a20_ikk': 0.3,      # A20 inhibition of IKK
            'ki_ikb_nfkb': 0.1,     # IkB inhibition of NFkB nuclear translocation
            
            # Initial conditions
            'tlr4_0': 1.0,
            'myd88_0': 1.0,
            'tak1_0': 0.0,
            'ikk_0': 1.0,
            'ikb_nfkb_0': 1.0,
            'nfkb_free_0': 0.0,
            'nfkb_nuclear_0': 0.0,
            'ikb_0': 0.2,
            'a20_0': 0.1,
            'inflammatory_genes_0': 0.0,
            'survival_genes_0': 0.0,
            'immune_genes_0': 0.0
        }
        
        # Store state variable names for easier access
        self.state_vars = [
            'TLR4', 'MyD88', 'TAK1', 'IKK', 'IkB_NFkB', 'NFkB_free', 
            'NFkB_nuclear', 'IkB', 'A20', 'inflammatory_genes', 
            'survival_genes', 'immune_genes'
        ]
    
    def update_parameters(self, new_params):
        """Update model parameters with new values."""
        self.params.update(new_params)
    
    def model_equations(self, t, y, lps_signal):
        """
        Define the system of ODEs for the signaling pathway.
        
        Parameters:
        t : float
            Current time point
        y : array
            Current state variables
        lps_signal : function
            Function that returns LPS concentration at time t
        
        Returns:
        dydt : array
            Derivatives of the state variables
        """
        # Extract state variables
        TLR4, MyD88, TAK1, IKK, IkB_NFkB, NFkB_free, NFkB_nuclear, IkB, A20, infl_genes, surv_genes, imm_genes = y
        
        # LPS stimulus (may be time-dependent)
        LPS = lps_signal(t)
        
        # Calculate derivatives
        dTLR4_dt = -self.params['k_lps_tlr4'] * LPS * TLR4 - self.params['gamma_tlr4'] * TLR4 + 0.01  # Small synthesis rate
        
        dMyD88_dt = (self.params['k_lps_tlr4'] * LPS * TLR4 - 
                     self.params['k_myd88_tak1'] * MyD88 - 
                     self.params['gamma_myd88'] * MyD88)
        
        # TAK1 activation with A20 inhibition
        dTAK1_dt = (self.params['k_myd88_tak1'] * MyD88 / (1 + self.params['ki_a20_tak1'] * A20) - 
                    self.params['k_tak1_ikk'] * TAK1 - 
                    self.params['gamma_tak1'] * TAK1)
        
        # IKK activation with A20 inhibition
        dIKK_dt = (self.params['k_tak1_ikk'] * TAK1 / (1 + self.params['ki_a20_ikk'] * A20) - 
                   self.params['k_ikk_ikb'] * IKK * IkB_NFkB - 
                   self.params['gamma_ikk'] * IKK)
        
        # IkB-NFkB complex dynamics
        dIkB_NFkB_dt = (-self.params['k_ikk_ikb'] * IKK * IkB_NFkB + 
                        self.params['k_nfkb_cytoplasmic'] * NFkB_nuclear * IkB - 
                        self.params['gamma_ikb'] * IkB_NFkB)
        
        # Free NFkB in cytoplasm
        dNFkB_free_dt = (self.params['k_ikk_ikb'] * IKK * IkB_NFkB - 
                         self.params['k_nfkb_nuclear'] * NFkB_free / (1 + self.params['ki_ikb_nfkb'] * IkB) - 
                         self.params['gamma_nfkb'] * NFkB_free)
        
        # Nuclear NFkB
        dNFkB_nuclear_dt = (self.params['k_nfkb_nuclear'] * NFkB_free / (1 + self.params['ki_ikb_nfkb'] * IkB) - 
                            self.params['k_nfkb_cytoplasmic'] * NFkB_nuclear * IkB - 
                            self.params['gamma_nfkb'] * NFkB_nuclear)
        
        # Free IkB synthesis and degradation
        dIkB_dt = (self.params['k_ikb_synthesis'] * NFkB_nuclear - 
                  self.params['k_nfkb_cytoplasmic'] * NFkB_nuclear * IkB - 
                  self.params['gamma_ikb'] * IkB)
        
        # A20 synthesis and degradation
        dA20_dt = (self.params['k_a20_synthesis'] * NFkB_nuclear - 
                   self.params['gamma_a20'] * A20)
        
        # Gene expression - with different dynamics for different gene types
        # Inflammatory genes - quick response
        dInfl_genes_dt = (self.params['k_gene_transcription'] * NFkB_nuclear * (1 + 2 * NFkB_nuclear / self.params['nfkb_nuclear_0']) - 
                         self.params['gamma_mrna'] * infl_genes)
        
        # Survival genes - sustained response
        dSurv_genes_dt = (0.8 * self.params['k_gene_transcription'] * NFkB_nuclear - 
                         0.7 * self.params['gamma_mrna'] * surv_genes)
        
        # Immune regulation genes - delayed response (I1-FFL like)
        delayed_activation = max(0, NFkB_nuclear - 0.5 * A20)
        dImm_genes_dt = (0.6 * self.params['k_gene_transcription'] * delayed_activation - 
                        0.9 * self.params['gamma_mrna'] * imm_genes)
        
        return [dTLR4_dt, dMyD88_dt, dTAK1_dt, dIKK_dt, dIkB_NFkB_dt, 
                dNFkB_free_dt, dNFkB_nuclear_dt, dIkB_dt, dA20_dt, 
                dInfl_genes_dt, dSurv_genes_dt, dImm_genes_dt]
    
    def simulate(self, t_span, t_eval=None, lps_stimulus=None):
        """
        Simulate the model over a time period.
        
        Parameters:
        t_span : tuple
            (t_start, t_end) simulation time span
        t_eval : array, optional
            Specific time points to evaluate
        lps_stimulus : function, optional
            Custom LPS stimulus function, default is step function
            
        Returns:
        t : array
            Time points
        y : array
            Solution array, each row is a state variable
        """
        # Default LPS stimulus is a step function
        if lps_stimulus is None:
            def lps_stimulus(t):
                return 1.0 if t > 0 else 0.0
        
        # Initial conditions
        y0 = [
            self.params['tlr4_0'],
            self.params['myd88_0'],
            self.params['tak1_0'],
            self.params['ikk_0'],
            self.params['ikb_nfkb_0'],
            self.params['nfkb_free_0'],
            self.params['nfkb_nuclear_0'],
            self.params['ikb_0'],
            self.params['a20_0'],
            self.params['inflammatory_genes_0'],
            self.params['survival_genes_0'],
            self.params['immune_genes_0']
        ]
        
        # Save the initial NFkB nuclear value for fold change calculations
        self.params['nfkb_nuclear_0'] = self.params['nfkb_nuclear_0'] or 0.01  # Avoid division by zero
        
        # Solve the ODE system
        solution = solve_ivp(
            lambda t, y: self.model_equations(t, y, lps_stimulus),
            t_span,
            y0,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        
        return solution.t, solution.y
    
    def plot_results(self, t, y, save_path=None):
        """
        Plot the simulation results.
        
        Parameters:
        t : array
            Time points
        y : array
            Solution array, each row is a state variable
        save_path : str, optional
            Path to save the figure
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        # Plot 1: LPS signaling cascade
        axs[0].plot(t, y[0], label='TLR4', linewidth=2)
        axs[0].plot(t, y[1], label='MyD88', linewidth=2)
        axs[0].plot(t, y[2], label='TAK1', linewidth=2)
        axs[0].plot(t, y[3], label='IKK', linewidth=2)
        axs[0].set_ylabel('Concentration (a.u.)')
        axs[0].set_title('LPS Signaling Cascade')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        # Plot 2: NFkB dynamics
        axs[1].plot(t, y[4], label='IkB-NFkB', linewidth=2)
        axs[1].plot(t, y[5], label='Free NFkB (cytoplasm)', linewidth=2)
        axs[1].plot(t, y[6], label='Nuclear NFkB', linewidth=2)
        axs[1].plot(t, y[7], label='Free IkB', linewidth=2)
        axs[1].plot(t, y[8], label='A20', linewidth=2)
        axs[1].set_ylabel('Concentration (a.u.)')
        axs[1].set_title('NFkB Pathway Dynamics')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        # Plot 3: Gene expression
        axs[2].plot(t, y[9], label='Inflammatory Genes', linewidth=2)
        axs[2].plot(t, y[10], label='Survival Genes', linewidth=2)
        axs[2].plot(t, y[11], label='Immune Regulation Genes', linewidth=2)
        axs[2].set_ylabel('Expression Level (a.u.)')
        axs[2].set_xlabel('Time (a.u.)')
        axs[2].set_title('Gene Expression Profiles')
        axs[2].legend()
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    def simulate_cell_population(self, n_cells=100, t_span=(0, 50), t_eval=None, parameter_cv=0.2):
        """
        Simulate a heterogeneous cell population with variable parameter values.
        
        Parameters:
        n_cells : int
            Number of cells to simulate
        t_span : tuple
            (t_start, t_end) simulation time span
        t_eval : array, optional
            Specific time points to evaluate
        parameter_cv : float
            Coefficient of variation for parameter sampling
            
        Returns:
        results : dict
            Dictionary containing simulation results for all cells
        """
        results = {
            'time': None,
            'nfkb_nuclear': [],
            'inflammatory_genes': [],
            'survival_genes': [],
            'immune_genes': [],
            'fold_changes': []
        }
        
        for i in range(n_cells):
            # Generate random parameters for this cell
            cell_params = {}
            for key, value in self.params.items():
                # Only vary rate constants, not initial conditions
                if key.startswith('k_') or key.startswith('gamma_') or key.startswith('ki_'):
                    # Log-normal distribution to ensure positive values
                    cell_params[key] = value * np.random.lognormal(sigma=parameter_cv)
            
            # Update model with this cell's parameters
            self.update_parameters(cell_params)
            
            # Simulate this cell
            t, y = self.simulate(t_span, t_eval)
            
            # Store results
            if results['time'] is None:
                results['time'] = t
            
            results['nfkb_nuclear'].append(y[6])
            results['inflammatory_genes'].append(y[9])
            results['survival_genes'].append(y[10])
            results['immune_genes'].append(y[11])
            
            # Calculate fold change of nuclear NFkB
            baseline = y[6][0] if y[6][0] > 0 else 0.01
            fold_change = y[6] / baseline
            results['fold_changes'].append(fold_change)
        
        return results
    
    def plot_population_results(self, results, save_path=None):
        """
        Plot the results from a cell population simulation.
        
        Parameters:
        results : dict
            Results from simulate_cell_population
        save_path : str, optional
            Path to save the figure
        """
        t = results['time']
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot nuclear NFkB for all cells
        for nfkb in results['nfkb_nuclear']:
            axs[0, 0].plot(t, nfkb, 'b-', alpha=0.1)
        
        # Calculate and plot the mean and standard deviation
        mean_nfkb = np.mean(results['nfkb_nuclear'], axis=0)
        std_nfkb = np.std(results['nfkb_nuclear'], axis=0)
        axs[0, 0].plot(t, mean_nfkb, 'r-', linewidth=2, label='Mean')
        axs[0, 0].fill_between(t, mean_nfkb - std_nfkb, mean_nfkb + std_nfkb, color='r', alpha=0.3)
        axs[0, 0].set_title('Nuclear NFkB Dynamics')
        axs[0, 0].set_xlabel('Time (a.u.)')
        axs[0, 0].set_ylabel('Concentration (a.u.)')
        axs[0, 0].legend()
        
        # Plot fold changes
        for fc in results['fold_changes']:
            axs[0, 1].plot(t, fc, 'g-', alpha=0.1)
        
        mean_fc = np.mean(results['fold_changes'], axis=0)
        std_fc = np.std(results['fold_changes'], axis=0)
        axs[0, 1].plot(t, mean_fc, 'r-', linewidth=2, label='Mean')
        axs[0, 1].fill_between(t, mean_fc - std_fc, mean_fc + std_fc, color='r', alpha=0.3)
        axs[0, 1].set_title('NFkB Fold Change')
        axs[0, 1].set_xlabel('Time (a.u.)')
        axs[0, 1].set_ylabel('Fold Change')
        axs[0, 1].legend()
        
        # Plot inflammatory genes
        for ig in results['inflammatory_genes']:
            axs[1, 0].plot(t, ig, 'r-', alpha=0.1)
        
        mean_ig = np.mean(results['inflammatory_genes'], axis=0)
        std_ig = np.std(results['inflammatory_genes'], axis=0)
        axs[1, 0].plot(t, mean_ig, 'k-', linewidth=2, label='Mean')
        axs[1, 0].fill_between(t, mean_ig - std_ig, mean_ig + std_ig, color='k', alpha=0.3)
        axs[1, 0].set_title('Inflammatory Gene Expression')
        axs[1, 0].set_xlabel('Time (a.u.)')
        axs[1, 0].set_ylabel('Expression Level (a.u.)')
        axs[1, 0].legend()
        
        # Plot immune regulation genes
        for irg in results['immune_genes']:
            axs[1, 1].plot(t, irg, 'b-', alpha=0.1)
        
        mean_irg = np.mean(results['immune_genes'], axis=0)
        std_irg = np.std(results['immune_genes'], axis=0)
        axs[1, 1].plot(t, mean_irg, 'k-', linewidth=2, label='Mean')
        axs[1, 1].fill_between(t, mean_irg - std_irg, mean_irg + std_irg, color='k', alpha=0.3)
        axs[1, 1].set_title('Immune Regulation Gene Expression')
        axs[1, 1].set_xlabel('Time (a.u.)')
        axs[1, 1].set_ylabel('Expression Level (a.u.)')
        axs[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Lab 8: Finding Motifs in Transcription Factor Networks
# Instructions for modeling and analysis of TLR4-NFkB signaling pathway

## 1. Introduction and Setup

# Let's start by creating our model and examining the TLR4-NFkB signaling pathway
# This lab builds on concepts covered in the lecture on network motifs and dynamics

print("Lab 8: Finding Motifs in Transcription Factor Networks")
print("Instructor: RK Azhigulova")
print("Date: March 24, 2025")
print("-------------------------------------------------------")

# Create model
model = TLR4_NFkB_Model()
print("Model initialized successfully.")

## 2. Single Cell Simulation with Continuous LPS Stimulation

# First, we'll simulate a single cell response to continuous LPS stimulation
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Define a constant LPS stimulus
def constant_lps(t):
    return 1.0 if t > 5 else 0.0

print("\nRunning single cell simulation with constant LPS stimulation...")
t, y = model.simulate(t_span, t_eval, constant_lps)

# Plot results
print("Generating plots for single cell response...")
model.plot_results(t, y)

## 3. Analysis of Different Motif Behaviors

print("\nThe model includes different types of network motifs:")
print("- Negative autoregulation (NAR) through IκB and A20 feedback")
print("- Incoherent feed-forward loop (I1-FFL) in immune regulation genes")
print("- Coherent feed-forward loop (C1-FFL) in inflammatory genes")

# Let's modify parameters to highlight different motif behaviors
print("\nComparing different network motif configurations...")

# Original parameters - save a copy
original_params = model.params.copy()

# Enhanced NAR configuration
model.update_parameters({
    'k_ikb_synthesis': 0.08,  # Increase IκB synthesis
    'k_a20_synthesis': 0.06,  # Increase A20 synthesis
})
t_nar, y_nar = model.simulate(t_span, t_eval, constant_lps)

# Reset and try enhanced I1-FFL configuration
model.update_parameters(original_params)
model.update_parameters({
    'ki_a20_tak1': 0.4,  # Increase inhibition strength in I1-FFL
})
t_i1ffl, y_i1ffl = model.simulate(t_span, t_eval, constant_lps)

# Reset parameters
model.update_parameters(original_params)

# Compare NFkB dynamics under different configurations
plt.figure(figsize=(10, 6))
plt.plot(t, y[6], 'b-', linewidth=2, label='Baseline')
plt.plot(t_nar, y_nar[6], 'r-', linewidth=2, label='Enhanced NAR')
plt.plot(t_i1ffl, y_i1ffl[6], 'g-', linewidth=2, label='Enhanced I1-FFL')
plt.xlabel('Time (a.u.)')
plt.ylabel('Nuclear NFkB (a.u.)')
plt.title('Effect of Different Network Motifs on NFkB Dynamics')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

## 4. Cell Population Analysis

# Now we'll simulate a heterogeneous cell population to examine variability
print("\nSimulating a heterogeneous cell population (n=50 cells)...")
results = model.simulate_cell_population(n_cells=50, t_span=t_span, t_eval=t_eval)

# Plot population results
print("Generating population analysis plots...")
model.plot_population_results(results)

## 5. Fold Change Detection Analysis

# Let's examine fold change behavior in our model by testing different LPS concentrations
print("\nTesting fold change detection with different LPS concentrations...")

lps_concentrations = [0.5, 1.0, 2.0]
fold_changes = []

plt.figure(figsize=(12, 8))

for i, lps_conc in enumerate(lps_concentrations):
    # Define constant LPS at different concentrations
    def lps_at_conc(t, concentration=lps_conc):
        return concentration if t > 5 else 0.0
    
    # Simulate
    t, y = model.simulate(t_span, t_eval, lambda t: lps_at_conc(t))
    
    # Plot NFkB and gene expression
    plt.subplot(2, 3, i+1)
    plt.plot(t, y[6], 'b-', linewidth=2)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Nuclear NFkB (a.u.)')
    plt.title(f'LPS = {lps_conc}')
    
    plt.subplot(2, 3, i+4)
    plt.plot(t, y[9], 'r-', linewidth=2, label='Inflammatory')
    plt.plot(t, y[10], 'g-', linewidth=2, label='Survival')
    plt.plot(t, y[11], 'b-', linewidth=2, label='Immune')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Gene Expression (a.u.)')
    if i == 0:
        plt.legend()

plt.tight_layout()
plt.show()

## 6. Conclusion and Lab Report Instructions

print("\nLab 8 Completion Tasks:")
print("1. Analyze how different network motifs affect signaling dynamics")
print("2. Explain the fold change detection behavior observed in the simulations")
print("3. Discuss cell-to-cell variability and its biological significance")
print("4. Submit a 2-page report with your analysis and conclusions")
print("\nReminder: Lab reports due by next week's session.")

