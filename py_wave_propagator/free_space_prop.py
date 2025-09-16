import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple
from scipy.interpolate import RegularGridInterpolator


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
                 sizePx: Tuple[float, float] = [6.45e-6, 6.45e-6], 
                 wl: float = 658*1e-9):
        """
        Class initialization. Base units in meters. 

        Args:
            numPx (Tuple[int, int], optional): Number of pixels [Nx, Ny]. Defaults to [1392, 1040].
            sizePx (Tuple[float, float], optional): Size of pixels [dx, dy]. Defaults to [6.45e-6, 6.45e-6].
            wl (float, optional): wavelength in air. Defaults to 658*1e-9.
        """
        
        ## setup params
        # wavelength, camera specs, resolutions, and limits

        ## Inputs
        self.wl = wl

        ## Camera-based calculations or planes size and resolution
        self.numPx = numPx # Pixels along the [x-axis, y-axis] or samples
        self.sizePx = [sizePx[0], sizePx[1]] # pixel size [x-axis, y-axis]

        self.sizeSensorX = self.numPx[0]*self.sizePx[0] # sensor size along x-axis
        self.sizeSensorY = self.numPx[1]*self.sizePx[1] # sensor size along y-axis

        # Sampling plane 2x for linearization
        self.Sx = 2*self.sizeSensorX
        self.Sy = 2*self.sizeSensorY

        self.samplingRate = 1/self.sizePx[0]

        # max freq to prevent aliasing by the transfer function
        self.delU = 1/self.Sx
        self.delV = 1/self.Sy

        ## -+ maxFreq = -+ samplingFreq/2
        self.freqRows = np.linspace(-1/(2*self.sizePx[0]), 1/(2*self.sizePx[0]), int(1/(self.sizePx[0]*self.delU)))
        self.freqCols = np.linspace(-1/(2*self.sizePx[1]), 1/(2*self.sizePx[1]), int(1/(self.sizePx[1]*self.delV)))

        self.u, self.v = np.meshgrid(self.freqRows, self.freqCols, indexing='ij')
        k = self.wl**(-2) - self.u**2 - self.v**2

        # removing evanscent waves: limiting transfer function does that but to remove 
        # numerical errors that may occur, safer to zero freqs > 1/wl
        if not np.all(k > 0):
            k[k < 0] = 0

        self.w = np.sqrt(k) # freq_z

        self.wavefield_z0 = None # wavefield at the input
        self.wavefield_z1 = None # wavefield at the output

        self.fft_wave_z0 = None # source plane spectrum with padding
        # self.fft_wave_z1 = None # parallel plane at dist z
        self.z = None # distance to propagate wavefield at z0 to parallel plane at z1

    
    def wavefield(self, wave: NDArray[np.complex128]):
        assert [wave.shape[0], wave.shape[1]] == self.numPx, "Incorrect number of pixels specified in constructor"
        # not using this wavefield to calculate here for speed as function may be used repeatedly

        linImg = np.zeros([int(self.Sy/self.sizePx[0]), int(self.Sx/self.sizePx[1])], dtype=np.complex128) # creates zeros of the size of the sensor
        linImg[int(linImg.shape[0]/2 - wave.shape[0]/2):int(linImg.shape[0]/2 + wave.shape[0]/2), 
            int(linImg.shape[1]/2 - wave.shape[1]/2):int(linImg.shape[1]/2 + wave.shape[1]/2)] = wave # ensures the wave is at the center of linImg
 
        self.wavefield_z0 = wave
        self.fft_wave_z0 = np.fft.fftshift(np.fft.fft2(linImg))
        # self.fft_wave_z0 = np.fft.fftshift(np.fft.fft2(wave, s=[wave.shape[0]*2, wave.shape[1]*2]))
        
    def propogate(self, dist: float):
        assert self.wavefield_z0 is not None, "Use method wavefied first"
        
        self.z = dist # distance to propogate along the z axis
        self.uLimit = 1/(np.sqrt((2*self.delU*self.z)**2 + 1)* self.wl)
        self.vLimit = 1/(np.sqrt((2*self.delV*self.z)**2 + 1)* self.wl)
        
        H = np.exp(1j*2*np.pi*self.w*self.z)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask = np.ones_like(H)
        mask[np.logical_or(np.abs(self.u) > int(self.uLimit), np.abs(self.v) >= int(self.vLimit))] = 0

        H = mask*H
        fft_wave_z1 = H*self.fft_wave_z0

        self.wavefield_z1 = np.fft.ifft2(np.fft.ifftshift(fft_wave_z1))
        self.wavefield_z1 = self.wavefield_z1[
            int(fft_wave_z1.shape[0]/2 - self.wavefield_z0.shape[0]/2):int(self.fft_wave_z0.shape[0]/2 + self.wavefield_z0.shape[0]/2),
            int(fft_wave_z1.shape[1]/2 - self.wavefield_z0.shape[1]/2):int(self.fft_wave_z0.shape[1]/2 + self.wavefield_z0.shape[1]/2)    
            ]
        
        return self.wavefield_z1
    
    def obliquePlaneProp(self, rotation : list = [0, 0], degrees: bool = True, samples_ref_spectrum: int = 512, shift: bool = False):

        # rotation applied on wavefield at z0
        # only works around for 1 axis at a time: see notebook v2_... for details
        
        assert self.wavefield_z0 is not None, "Use method wavefied first"
        
        if degrees:
            rot_x = rotation[0]*np.pi/180
            rot_y = rotation[1]*np.pi/180
        else:
            rot_x = rotation[0]
            rot_y = rotation[1]

        Nx = self.wavefield_z0.shape[0]
        Ny = self.wavefield_z0.shape[1]
        
        u_hat = np.fft.fftshift(np.fft.fftfreq(Nx, self.sizePx[0]))
        v_hat = np.fft.fftshift(np.fft.fftfreq(Ny, self.sizePx[1]))
        
        fft_wave_z0 = np.fft.fftshift(np.fft.fft2(self.wavefield_z0))

        interp = RegularGridInterpolator((u_hat, v_hat), fft_wave_z0, method='linear', bounds_error=False, fill_value=0.)
        # interp_re = RegularGridInterpolator((u_hat, v_hat), fft_wave_z0.real, method='linear', bounds_error=False, fill_value=0.)
        # interp_im = RegularGridInterpolator((u_hat, v_hat), fft_wave_z0.imag, method='linear', bounds_error=False, fill_value=0.)

        T_inv_x = np.array([[1, 0, 0], 
                            [0, np.cos(rot_x), -1*np.sin(rot_x)], 
                            [0, np.sin(rot_x), np.cos(rot_x)]], dtype=np.float64)

        T_inv_y = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], 
                            [0, 1, 0], 
                            [-1*np.sin(rot_y), 0, np.cos(rot_y)]], dtype=np.float64)

        T_inv = np.matmul(T_inv_y, T_inv_x)
        
        u_max = 1/(2*self.sizePx[0]) # max source plane
        v_max = 1/(2*self.sizePx[1]) # max source plane
        w_max = np.sqrt(1/(self.wl**2) - u_max**2 - v_max**2)
        
        UVW_hat_max = np.matmul(np.matmul(T_inv_x.transpose(), T_inv_y.transpose()), np.array([u_max, v_max, w_max]).reshape(3, 1))
        UVW_hat_min = np.matmul(np.matmul(T_inv_x.transpose(), T_inv_y.transpose()), np.array([-1*u_max, -1*v_max, w_max]).reshape(3, 1))
        
        UVW_hat_shift = -1*np.matmul(np.matmul(T_inv_x.transpose(), T_inv_y.transpose()), np.array([0, 0, 1/self.wl]).reshape(3, 1))

        u_hat = np.linspace(UVW_hat_min[0], UVW_hat_max[0], samples_ref_spectrum)
        v_hat = np.linspace(UVW_hat_min[1], UVW_hat_max[1], samples_ref_spectrum)
        
        U_hat, V_hat = np.meshgrid(u_hat, v_hat, indexing='ij')
        W_hat = np.sqrt(1/((self.wl)**2) - U_hat**2 - V_hat**2)

        UVW = np.matmul(T_inv, np.stack([U_hat, V_hat, W_hat], axis=0).reshape(3, -1)).reshape(3, *U_hat.shape)
        U, V, W = UVW
        
        J = np.abs((T_inv[0, 1]*T_inv[1, 2] - T_inv[0, 2]*T_inv[2, 1])*U_hat/W_hat + (T_inv[0, 2]*T_inv[1, 0] - T_inv[0, 0]*T_inv[1, 2])*V_hat/W_hat + T_inv[0, 0]*T_inv[1, 1] - T_inv[0, 1]*T_inv[1, 0])
        wave_z_obl = np.fft.ifft2(np.fft.ifftshift(interp((U, V))*J))[:Nx, :Ny] # needs more consideration
        # wave_z_obl = np.fft.ifft2(np.fft.ifftshift((interp_re((U, V)) + 1j*interp_im((U, V)))*J))
        
        cx = Nx/(UVW_hat_max[0] - UVW_hat_min[0])[0]
        cy = Ny/(UVW_hat_max[1] - UVW_hat_min[1])[0]
        x = np.linspace(0, cx, Nx)
        y = np.linspace(0, cy, Ny)

        interp = RegularGridInterpolator((x, y), wave_z_obl, method='linear', bounds_error=False, fill_value=0.)
        
        x = np.linspace(cx/2 - Nx/2*self.sizePx[0], cx/2 + Nx/2*self.sizePx[0], Nx)
        y = np.linspace(cy/2 - Ny/2*self.sizePx[1], cy/2 + Ny/2*self.sizePx[1], Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        wave_z_obl_2 = interp((X, Y))
        
        if shift:
            wave_z_obl_2 = wave_z_obl_2*np.exp(1j*2*np.pi*(UVW_hat_shift[0]*X + UVW_hat_shift[1]*Y))
        
        return wave_z_obl_2

    def setup_limit_info(self):
        # TBD, this may not be accurate.
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

        print(f'Max Freq and Angle the camera/plane can record without aliasing: {maxFreqPossible} cycles/m | {maxAnglePossible} radians')
        
        print(f'Max Freq and Angle the setup allows (freqs > do not reach the next plane): {(maxFreqSetupX, maxFreqSetupY)} cycles/m | {(maxAngleSetupX, maxAngleSetupY)} radians')

    
    def visualizations(self):
        """
        Too many repeated lines of code in the main scripts.
        Should be able to plot the spectrums with correct ticks.
        """

        pass