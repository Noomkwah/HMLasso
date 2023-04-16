
## FOR USERS

Lasso regularization that performs well with high missing rate.

Implemented according to the related article 'HMLasso: Lasso with High Missing
Rate' by Masaaki Takada1, Hironori Fujisawa and Takeichiro Nishikawa.
Link to the article: https://www.ijcai.org/proceedings/2019/0491.pdf

------------
Common uses: Once fitted, the HMLasso can provide linear predictions. 
It can also be used to select variables of interest from the given data. This 
second goal can be achieved through selection of variables whose coefficient
is almost (or equal to) zero.

Please note that no metric is implemented in this class for now. 
See sklearn.metrics.mean_squared_error or like for useful metrics.

  ------------
Common error: During the fitting HMLasso.fit(X, y), errors such as
'ArpackNoConvergence: ARPACK error -1: No convergence' may occur. It comes 
from the fact that the underlying solver used did not successfully assess
the positive-semidefiniteness of the inner variable Sigma_opt. If you are 
sure that Sigma_opt is PSD (which is likely to be the case in normal uses of 
the estimator), you can add the two following lines:
"
from file_04_HMLasso import ERRORS_HANDLING
ERRORS_HANDLING = 'ignore'
"
If the problem persists, then maybe praying god is the only remaining thing
you can do.


## FOR THE AUTHOR, FOR DEVELOPERS

When using "ERRORS_HANDLING = 'ignore'", what happens is that the PSD check of Sigma_opt is skipped, and the solver try to optimizer the objective without checking whether or not it is possible to do so. It is therefore highly unrecommanded to set ERRORS_HANDLING to 'ignore' if you are able not to do it.

Other curious phenomenon: due to floating points exceptions, Sigma_opt is not always PSD. This problem is different from the previously mentioned one as, in our current case, Sigma_opt is actually NOT PSD (while it may have been in the previous problem. The solver being unable to check whether or not it was). The HMLasso class handle this issue by artificially increasing all eigenvalues of Sigma_opt until the least one is positive. 

This is what this code do:
"
eigenvalues = np.linalg.eig(self.Sigma_opt)[0]
    min_eigenvalue = min(eigenvalues)
    if min_eigenvalue < 0:
      print(f"[Warning] Sigma_opt is not PSD, its minimum eigenvalue is {min_eigenvalue}. Error handled by adding {-min_eigenvalue} to each eigenvalue.")
      self.Sigma_opt = self.Sigma_opt - min_eigenvalue * np.eye(self.p, self.p)
"
However, it appears that (I am not sure if this trully comes from these lines) doing so might transform Sigma_opt into a complex-valued matrix, with all coefficients being things like "5.25 + 0.0j" or like. To fix this issue, I added the line "self.Sigma_opt = np.real(self.Sigma_opt)" that just drops the imaginary part of the matrix. It should not cause any problem, but I admit this way of proceeding is a bit dirty...
