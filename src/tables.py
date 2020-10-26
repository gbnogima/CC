from os import makedirs
from evaluate import evaluate_directory, statistical_significance, statistical_significance_CC
import settings
from quapy.error import mae, mrae, ae, RAE

result_path = '../results'
tables_path = '../tables'
MAXTONE = 50  # sets the intensity of the maximum color reached by the worst (red) and best (green) results
makedirs(tables_path, exist_ok=True)

sample_length=settings.SAMPLE_SIZE
methods=['cc', 'acc', 'pcc', 'pacc']
learners = ['svm', 'lr', 'rf', 'mnb', 'cnn']
datasets = ['imdb', 'kindle', 'hp']
optimization_measures = ['none', 'acce', 'f1e', 'mae']
evaluation_measures = [mae, mrae]

results_dict = evaluate_directory('../results/*.pkl', evaluation_measures)
stats = {
    'hp': {
        'mae': statistical_significance('../results/hp-*ae-run?.pkl', ae),
        'mrae': statistical_significance('../results/hp-*ae-run?.pkl', RAE),
    },
    'kindle': {
        'mae': statistical_significance('../results/kindle-*ae-run?.pkl', ae),
        'mrae': statistical_significance('../results/kindle-*ae-run?.pkl', RAE),
    },
    'imdb': {
        'mae': statistical_significance('../results/imdb-*ae-run?.pkl', ae),
        'mrae': statistical_significance('../results/imdb-*ae-run?.pkl', RAE),
    }
}

nice = {
    'none':'\O',
    'acce':'A',
    'f1e':'F_1',
    'mae':'AE',
    'mrae':'RAE',
    'svmkld': 'SVM(KLD)',
    'svmnkld': 'SVM(NKLD)',
    'svmq': 'SVM(Q)',
    'svmae': 'SVM(AE)',
    'svmnae': 'SVM(NAE)',
    'svmmae': 'SVM(AE)',
    'svmmrae': 'SVM(RAE)',
    'quanet': 'QuaNet',
    'hdy': 'HDy',
    'svmperf':''#''\\textit{perf}'
}

def nicerm(key):
    return '\mathrm{'+nice[key]+'}'

def color_from_rel_error(rel_error):
    color = 'red' if sign == '+' else 'green'
    tone = int(MAXTONE*abs(rel_error)/100)
    return '\cellcolor{' + color + f'!{min(tone, MAXTONE)}' + '}'

# first batch of 4 tables (CC, ACC, PCC, and PACC)
# --------------------------------------------------------
print(f'Creating tables for CC variants:')
for method in methods:
    latex_path = f'{tables_path}/{method.upper()}.tex'
    print(f'\twriting table {latex_path} for method {method}')

    if method == 'cc':
        caption = """Results showing how the quantification error of CC changes 
        according to the measure used in hyperparameter optimization; a 
        negative percentage indicates a reduction in error with respect to 
        using the method with default parameters. The background cell color
        indicates improvement (green) or deterioration (red), while its 
        tone intensity is proportional to the absolute magnitude. """
    else:
        caption = 'Same as Table~\\ref{tab:CC}, but with '+method.upper()+' instead of CC.'


    table ="\\begin{table}[t]\caption{"+caption+"}\label{tab:"+method.upper()+"} \\resizebox{\\textwidth}{!} {"

    tabular="""
    \\begin{tabular}{|l||ll|ll||ll|ll||ll|ll|}
    \hline
    & \multicolumn{4}{c||}{\\textsc{IMDB}} 
    & \multicolumn{4}{c||}{\\textsc{Kindle}} 
    & \multicolumn{4}{c|}{\\textsc{HP}} \\\\ 
    \hline
    & \multicolumn{2}{c}{AE} 
    & \multicolumn{2}{|c||}{RAE} 
    & \multicolumn{2}{c}{AE} 
    & \multicolumn{2}{|c||}{RAE} 
    & \multicolumn{2}{c}{AE} 
    & \multicolumn{2}{|c|}{RAE} \\\\
    \hline
    """

    for learner in learners:
        for optimization_measure in optimization_measures:
            tabular += method.upper()+'$^{\mathrm{'+nice[optimization_measure]+'}}_{\mathrm{'+learner.upper()+'}}$ '
            for dataset in datasets:

                for evaluation_measure in evaluation_measures:
                    result = f'{dataset}-{method}-{learner}-{sample_length}-{optimization_measure}-{evaluation_measure.__name__}'
                    if result in results_dict:
                        score = results_dict[result]
                        if optimization_measure != 'none':
                            ref = results_dict[f'{dataset}-{method}-{learner}-{sample_length}-none-{evaluation_measure.__name__}']
                            rel_error_reduction = -100*(ref-score)/ref
                            sign = '+' if rel_error_reduction>0 else '-'
                            cellcolor = color_from_rel_error(rel_error_reduction)
                            tabular += '& '+cellcolor+f'{score:.3f} & ' + cellcolor +f' ({sign}{abs(rel_error_reduction):.1f}\%)'
                        else:
                            tabular += f'& {score:.3f} & '

                    else:
                        tabular += '& \\textbf{---} & '
                        #print(f'{result}: {evaluation_measure.__name__}={score:.3f}')
            tabular+='\\\\\n\t'
        tabular+='\hline\n\n\t'
    tabular += '\end{tabular}\n'

    with open(latex_path, 'wt') as foo:
        table += tabular
        table += "}\n\end{table}\n"
        foo.write(table)


# final summary table
# --------------------------------------------------------
print(f'Creating overview table:')
table = """
\\begin{table}[t]
  \caption{Results showing how CC and its variants, once optimised
  using a quapy measure, compare with state-of-the-art
  baselines. \\textbf{Boldface} indicates the best method. 
  Superscripts $\dag$ and $\dag\dag$ denote the
  method (if any) whose score is not statistically significantly
  different from the best one according to a paired sample, two-tailed 
  t-test at different confidence levels: symbol $\dag$ indicates 
  $0.001<p$-value$<0.05$ while symbol $\dag\dag$ indicates 
  $0.05\leq p$-value. The absence of any such symbol indicates
  $p$-value $\leq 0.001$.
  }
  \label{tab:overview}
  \center
"""
#\\resizebox{\\textwidth}{!} {

tabular = """
  \\begin{tabular}{|l|l||r|r||r|r||r|r|}
    \hline
    \multicolumn{2}{|c||}{\mbox{}}
    & \multicolumn{2}{c||}{\\textsc{IMDB}} 
    & \multicolumn{2}{c||}{\\textsc{Kindle}} 
    & \multicolumn{2}{c|}{\\textsc{HP}} \\\\ 
    \cline{3-8}
    \multicolumn{2}{|c||}{\mbox{}}
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c||}{RAE} 
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c||}{RAE} 
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c|}{RAE}  \\\\
    \hline
    \multirow{8}{*}{\\begin{sideways}Baselines\end{sideways}}
    """

def color_from_rel_rank(rel_rank, maxtone=100):
    rel_rank = rel_rank*2-1
    if rel_rank < 0:
        color = 'red'
        tone = maxtone*(-rel_rank)
    else:
        color = 'green'
        tone = maxtone*rel_rank
    return '\cellcolor{' + color + f'!{int(tone)}' + '}'

def add_result(method, learner, optimization_measure):
    tabular = ""
    tabular += ' & ' + nice.get(method, method.upper()) + \
               '$^{\mathrm{' + nice.get(optimization_measure, optimization_measure.upper()) + '}}' + \
               '_{\mathrm{' + nice.get(learner, learner.upper()) + '}}$ '
    for dataset in datasets:
        for evaluation_measure in evaluation_measures:
            evalname = evaluation_measure.__name__
            result = f'{dataset}-{method}-{learner}-{sample_length}-{optimization_measure}-{evalname}'
            if result in results_dict:
                score = results_dict[result]
                stat_dic = stats[dataset][evalname]
                method_key = f'{dataset}-{method}-{learner}-{sample_length}-{optimization_measure}'
                if method_key not in stat_dic:
                    if method!='mlpe':
                        print(f'warning: method {method_key} is not part of the statistical significance test study')
                    stat_sig, rel_rank = ('verydifferent', 0)
                else:
                    stat_sig, rel_rank = stat_dic[method_key]

                if stat_sig=='best':
                    tabular += '& \\textbf{'+ f'{score:.3f}'+'}' + '$\phantom{\dag}\phantom{\dag}$'
                elif stat_sig=='verydifferent':
                    tabular += f'& {score:.3f}' + '$\phantom{\dag}\phantom{\dag}$'
                elif stat_sig=='different':
                    tabular += f'& {score:.3f}'+'$\dag\phantom{\dag}$'
                elif stat_sig=='nondifferent':
                    tabular += f'& {score:.3f}'+'$\dag\dag$'
                else:
                    print('stat sig error: ' + stat_sig)
                tabular += color_from_rel_rank(rel_rank, maxtone=MAXTONE)
            else:
                tabular += '& \\textbf{---} '
    tabular += '\\\\\n\t'
    return tabular

tabular += add_result('emq', 'lr', 'mae')
tabular += add_result('svmkld', 'svmperf', 'mae')
tabular += add_result('svmnkld', 'svmperf', 'mae')
tabular += add_result('svmq', 'svmperf', 'mae')
tabular += add_result('svmmae', 'svmperf', 'mae')
#tabular += add_result('svmmae', 'svmperf', 'mrae')
#tabular += add_result('svmmrae', 'svmperf', 'mae')
tabular += add_result('svmmrae', 'svmperf', 'mrae')
#tabular += add_result('hdy', 'lr', 'none')
tabular += add_result('hdy', 'lr', 'mae')
#tabular += add_result('ave', 'svr', 'none')
#tabular += add_result('quanet', 'cnn', 'none')
tabular += add_result('quanet', 'cnn', 'mae')
#tabular += '\\hline'
tabular += '\cline{2-8}\n'
tabular += add_result('mlpe', 'none', 'none')


"""
& SVM(KLD)$^{\mathrm{KLD}}$ & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) \\\\
& SVM(NKLD)$^{\mathrm{NKLD}}$ & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) \\\\
& SVM(Q)$^{\mathrm{Q}}$ & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) \\\\
& SVM(AE)$^{\mathrm{AE}}$ & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) \\\\
& SVM(NAE)$^{\mathrm{NAE}}$ & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) \\\\
& QuaNet$^{\mathrm{AE}}_{\mathrm{LSTM}}$ & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) & 0.000 & (-00.0\%) \\\\
"""

tabular += """
    \hline\hline
    \multirow{20}{*}{\\begin{sideways}CC and its variants\end{sideways}}"""

for learner in learners:
    for method in methods:
        tabular += add_result(method, learner, 'mae')
    tabular += '\cline{2-8}\n'

tabular += """
    \hline
  \end{tabular}
  """

with open(f'{tables_path}/overview.tex', 'wt') as foo:
    table += tabular
    table += "\n\end{table}\n"
    foo.write(table)


# statistical significance test table for CC variants
# --------------------------------------------------------
print(f'Creating Statistical Significance Test table for the CC variants:')

metrics = [ae, RAE]
ss = {ccx: {m:statistical_significance_CC(ccx, m, datasets, learners, optimization_measures) for m in metrics} for ccx in methods}

table = """
\\begin{table}[t]
  \caption{
  Two-sided t-test results on \emph{related} samples of error scores across datasets and learners.
  For a pair of optimization measures X vs Y, symbol $\gg$ (resp. $>$) indicates that method X performs
  better (i.e., yields lower error) than Y, and that the difference in performance, as averaged across 
  pairs of experiments through datasets 
  and learners, is statistically significant at a confidence score of $\\alpha=0.001$ (resp. $\\alpha=0.05$).
  Symbols $\ll$ and $<$ hold similar meaning but indicate X performs worse (i.e., yields higher error) than Y.
  Symbol $\sim$ instead indicates the differences in performance between X and Y are not
  statistically significantly different, i.e., that $p$-value $>= 0.05$. 
  }
  \label{tab:stats}
  \center
"""

tabular = """
  \\begin{tabular}{|rcl||c|c||c|c||c|c||c|c|}
    \hline
    \multicolumn{3}{|c||}{\mbox{}}
    & \multicolumn{2}{c||}{\\textsc{CC}} 
    & \multicolumn{2}{c||}{\\textsc{ACC}} 
    & \multicolumn{2}{c|}{\\textsc{PCC}} 
    & \multicolumn{2}{c|}{\\textsc{PACC}}\\\\ 
    \cline{4-11}
    \multicolumn{3}{|c||}{\mbox{}}
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c||}{RAE} 
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c||}{RAE} 
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c|}{RAE}
    & \multicolumn{1}{c|}{AE} 
    & \multicolumn{1}{c|}{RAE} \\\\
    \hline\n
    """
optims = optimization_measures[::-1]
for i, optim_i in enumerate(optims):
    for optim_j in optims[i+1:]:
        tabular += f'${nicerm(optim_i)}$ & vs & ${nicerm(optim_j)}$'
        for method in methods:
            for metric in metrics:
                tabular += ' & ' + ss[method][metric][f'{optim_i}-{optim_j}']
        tabular += '\\\\\n'
tabular += """
    \hline
  \end{tabular}
  """
with open(f'{tables_path}/ttest.tex', 'wt') as foo:
    table += tabular
    table += "\n\end{table}\n"
    foo.write(table)

print("[Done]")