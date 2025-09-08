import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple


class Wave2d:
    """
    Given a wavefield in a plane (x, y, 0) and other optical parameters,
    calculates the wave components at another plane. Propoagation to angled
    surfaces supported.
    
    1. Matsushima, K. and Shimobaba, T., 2009. Band-limited angular spectrum method 
    for numerical simulation of free-space propagation in far and near fields. 
    Optics express, 17(22), pp.19662-19673.
    
    2. Matsushima, K., Schimmel, H. and Wyrowski, F., 2003. Fast calculation method 
    for optical diffraction on tilted planes by use of the angular spectrum of plane 
    waves. Journal of the Optical Society of America A, 20(9), pp.1755-1762.
    """

    def __init__(self, 
                 numPx: Tuple[int, int] = [1392, 1040], 
                 sizePx: Tuple[float, float] = [0.00645, 0.00645], 
                 wl: float = 658*1e-6):
        """
        Class initialization. Base units in mm. 

        Args:
            numPx (Tuple[int, int], optional): Number of pixels [Nx, Ny]. Defaults to [1392, 1040].
            sizePx (Tuple[float, float], optional): Size of pixels [dx, dy]. Defaults to [0.00645, 0.00645].
            wl (float, optional): wavelength in air. Defaults to 658*1e-6.
        """
        
        ## setup params
        # wavelength, camera specs, resolutions, and limits

        self.mm = 1e-3 # standard otherwise specified
        ## Inputs
        self.wl = wl*self.mm

        ## Camera-based calculations or planes size and resolution
        self.numPx = numPx # Pixels along the [x-axis, y-axis] or samples
        self.sizePx = [sizePx[0]*self.mm, sizePx[1]*self.mm] # pixel size [x-axis, y-axis]

        self.sizeSensorX = self.numPx[0]*self.sizePx[0] # sensor size along x-axis
        self.sizeSensorY = self.numPx[1]*self.sizePx[1] # sensor size along y-axis

        # Sampling plane 2x for linearization
        self.Sx = 2*self.sizeSensorX
        self.Sy = 2*self.sizeSensorY

        self.samplingRate = 1/self.sizePx[0]

        # max freq to prevent aliasing by the transfer function
        self.delU = 1/self.Sx
        self.delV = 1/self.Sy

        self.freqRows = np.linspace(-1/(2*self.sizePx[0]), 1/(2*self.sizePx[0]), int(1/(self.sizePx[0]*self.delU))) ## -maxFreq/2, +maxFreq/2
        self.freqCols = np.linspace(-1/(2*self.sizePx[1]), 1/(2*self.sizePx[1]), int(1/(self.sizePx[1]*self.delV)))

        self.u, self.v = np.meshgrid(self.freqRows, self.freqCols)
        k = self.wl**(-2) - self.u**2 - self.v**2

        # removing evanscent waves: limiting transfer function does that but to remove 
        # numerical errors that may occur, safer to zero freqs > 1/wl
        if not np.all(k > 0):
            k[k < 0] = 0

        self.w = np.sqrt(k) # freq_z

        self.wavefield_z0 = None
        self.wavefield_z1 = None

        self.fft_wave_z0 = None
        self.fft_wave_z1 = None
        self.z = None # distance to propagate wavefield at z0 to parallel plane at z1

        # same optical axis as self.wavefield_z1 and self.wavefield_z0 at z1
        # self.uOblique = np.copy(self.u) # w/o x and y ticks spectrum at z1 oblique should look the same as at not-oblique
        # self.vOblique = np.copy(self.v)
    
    def propogate(self, dist: float):
        assert self.wavefield_z0 is not None, "Use method wavefied first"
        assert self.fft_wave_z0 is not None, "Use method wavefied first"

        self.z = dist*self.mm # distance to propogate along the z axis
        self.uLimit = 1/(np.sqrt((2*self.delU*self.z)**2 + 1)* self.wl)
        self.vLimit = 1/(np.sqrt((2*self.delV*self.z)**2 + 1)* self.wl)
        
        H = np.exp(1j*2*np.pi*self.w*self.z)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask = np.ones_like(H)
        mask[np.logical_or(np.abs(self.u) > int(self.uLimit), np.abs(self.v) >= int(self.vLimit))] = 0

        H = mask*H
        self.fft_wave_z1 = H*self.fft_wave_z0

        self.wavefield_z1 = np.fft.ifft2(np.fft.fftshift(self.fft_wave_z1))
        self.wavefield_z1 = self.wavefield_z1[
            int(self.fft_wave_z1.shape[0]/2 - self.wavefield_z0.shape[0]/2):int(self.fft_wave_z0.shape[0]/2 + self.wavefield_z0.shape[0]/2),
            int(self.fft_wave_z1.shape[1]/2 - self.wavefield_z0.shape[1]/2):int(self.fft_wave_z0.shape[1]/2 + self.wavefield_z0.shape[1]/2)    
            ]
        
        return self.wavefield_z1
    
    def obliquePlaneFreqs(self, rotation : list = [0, 0]):
        
        # waves received at z1: a plane perfectly orthogonal to the optical axis aka parallel plane
        # object at z0 taken to not be parallel
        # once propagated back from z1 to a porallel plane at z0, use fn to evaluate 
        # freqs at the non-prallel plane at z0
        # Note: freqs coefficients and wavefield coefficient remain exactly the same (exxcluding J(u, v)), 
        # only  the freq bins shift based on the tilt of the plane in space.

        # rotation theorem: apply for plane XZ and plane YZ

        assert self.fft_wave_z1 is not None, "Use method propagate first" 
        # if oblique on z=0, then propagate to with dist = 0 so as to not change the 
        # input wavefield and corresponding fft
        
        theta = rotation[0]
        phi = rotation[1]

        # uOblique = np.copy(self.u)
        # vOblique = np.copy(self.v)
        fz = self.wl**(-2) - self.u**2 - self.v**2
        # removing evanscent waves: limiting transfer function does that but to remove 
        # numerical errors that may occur, safer to zero freqs > 1/wl
        if not np.all(fz > 0):
            fz[fz < 0] = 0
        
        fz = np.sqrt(fz)

        T_inv = np.array([[np.cos(phi), 0, np.sin(phi)], 
                          [0, 1, 0], 
                          [-1*np.sin(phi), 0, np.cos(phi)]]) # source to ref plane transformation
        
        T = np.linalg.inv(T_inv) # ref to source plane transformation

        print(f'Rotation aroud (X-axis, Y-axis): {(theta, phi)}')

        # a, b = np.cos(theta), np.sin(theta)
        # c, d = np.cos(phi), np.sin(phi)

        if theta != 0: # rotation around x
            # uOblique = uOblique
            # vOblique = vOblique*a - fz*b
            # fz = vOblique*b + fz*a
            pass
        
        if phi != 0: # rotation around y
            # uOblique = uOblique*c + fz*d
            # vOblique = vOblique
            # fz = -1*uOblique*d + fz*c
            uOblique = T[0, 0]*self.u + T[0, 1]*self.v + T[0, 2]*fz
            vOblique = T[1, 0]*self.u + T[1, 1]*self.v + T[1, 2]*fz
            fzOblique = T[2, 0]*self.u + T[2, 1]*self.v + T[2, 2]*fz

            # uOblique = uOblique*T_inv[0, 0] + fzOblique*T_inv[0, 2]

            #Jacobian when rotation around y
            J_phi = T_inv[0, 0] - (uOblique/fzOblique)*T_inv[0, 2]

        return [uOblique, vOblique, J_phi]

    def wavefield(self, wave: NDArray[np.complex128]):
        assert [wave.shape[1], wave.shape[0]] == self.numPx, "Incorrect number of pixels specified in constructor"
        # not using this wavefield to calculate here for speed as function may be used repeatedly

        linImg = np.zeros([int(self.Sy/self.sizePx[1]), int(self.Sx/self.sizePx[0])], dtype=np.complex128) # creates zeros of the size of the sensor
        linImg[int(linImg.shape[0]/2 - wave.shape[0]/2):int(linImg.shape[0]/2 + wave.shape[0]/2), 
            int(linImg.shape[1]/2 - wave.shape[1]/2):int(linImg.shape[1]/2 + wave.shape[1]/2)] = wave # ensures the wave is at the center of linImg
 
        self.wavefield_z0 = wave
        self.fft_wave_z0 = np.fft.fftshift(np.fft.fft2(linImg))

    def setup_limit_info(self):
        assert self.z != None, "Distance is set to none"
        ## Nice to know 1: max freq and angle that the camera/plane can record without aliasing
        maxFreqPossible = self.samplingRate/2
        maxAnglePossible = self.samplingRate*self.wl # cosine

        ## Nice to know 2: max freq and angle that the setup allows. Freqs above these will not
        ## reach the next plane
        maxAngleSetupX = self.sizeSensorX/(2*np.linalg.norm([self.z, self.sizeSensorX])) # cosine \thetaX
        maxAngleSetupY = self.sizeSensorY/(2*np.linalg.norm([self.z, self.sizeSensorY])) # cosing \thetaY

        maxFreqSetupX = maxAngleSetupX/self.wl
        maxFreqSetupY = maxAngleSetupY/self.wl

        print(f'Max Freq and Angle the camera/plane can record without aliasing: {maxFreqPossible*self.mm} cycles/mm | {maxAnglePossible} radians')
        
        print(f'Max Freq and Angle the setup allows (freqs > do not reach the next plane): {(maxFreqSetupX*self.mm, maxFreqSetupY*self.mm)} cycles/mm | {(maxAngleSetupX, maxAngleSetupY)} radians')

    
    def visualizations(self):
        """
        Too many repeated lines of code in the main scripts.
        Should be able to plot the spectrums with correct ticks.
        """

        pass