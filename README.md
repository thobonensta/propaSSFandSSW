# propaSSFandSSW
A repository with code for the local version of SSW and also an implementation of SSF to validate the results. 
This version is with refraction (free-space, linear and trilinear profile) and relief (simple relief). It is
based on the discrete wide-angle SSF method [1] and the local version of SSW [2]. We also added the continuous
counterpart of SSF (narrow-angle and wide-angle) for comparison. 

# Git command
To clone the repository in order to have a branch to date and be able to modify the command in terminal is
> git clone https://github.com/thobonensta/propaSSFandSSW.git

This will create the associated repository in the file you are. Then you can modify everything and when you
want you can add, commit and push. If you create a new file, you have to
> git add filename

If you modify or add or remove you will then have to
> git commit -m "what did you do"

After that you can push the modifications
> git push 

If you are using PyCharmPro it can be done directlt through the interface of PyCharm.

# Documentation

propaSSFandSSW is comprised of 6 repository and one main file.
<ul>
<li> <strong>utilsSSW</strong> which contains all the necessary functions to perform the local SSW method</li>
<li> <strong>utilsRefraction</strong> which contains the code for the linear and trilinear refractive profile</li>
<li> <strong>utilsRelief</strong> that contains the functions to perform the staircaze model and then the translations</li>
<li> <strong>utilsSource</strong> which contains the file to compute the initial field</li>
<li> <strong>utilsSpace</strong> that contains the operators that are performed in the spatial domain</li>
<li> <strong>utilsSSF</strong> which contains all the necessary function for SSF (discrete wide angle but also continuous if wanted)</li>
</ul>	

## References
[1] Zhou, H., Chabory, A., & Douvenot, R. (2017). A 3-D split-step Fourier algorithm based on a discrete spectral representation of the propagation equation. IEEE Transactions on Antennas and Propagation, 65(4), 1988-1995. 

[2] Bonnafont, T., Douvenot, R., & Chabory, A. (2021). A local split-step wavelet method for the long range propagation simulation in 2D. Radio science, 56(2), 1-11.




