import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

from matplotlib import pyplot as plt


def propagate_beam_vol(
    field: NDArray[np.complex128], 
    RI_distribution: NDArray[np.float64], 
    RI_background: float, 
    wavelength: float, 
    spatial_resolution: Tuple[float, float, float], 
    padding: Optional[int] = None
    )-> NDArray[np.complex64]:
    """
    Propagates a 2D complex wavefield using the beam propagationm method. Base units is meters.

    Args:
        field (NDArray[np.complex128]): Input 2D complex wavefield in the (x, y, 0) plane
        RI_distribution (NDArray[np.float64]): 3D refractive index distribution (x, y, z)
        RI_background (float): background refractive index
        wavelength (float): wavelength of source in vacuum
        spatial_resolution (Tuple[float, float, float]): (dx, dy, dz). dz is the propagation step or slice thickness
        padding (Optional[int], optional): Number of pixels added to the field. Defaults to None.

    Returns:
        NDArray[np.complex64]: 2D complex wavefield at (x, y, -1)
    """
    
    if padding:
        field = np.pad(field, padding, 'edge') # edge make sense
        # symmteric would add diffraction pattern from outside FOV
        # zeros will make the aperture diffraction pattern dominates the diffraction patter after a certain distance
        
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy, dz = spatial_resolution
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')

    Kz = np.sqrt(0j + (k0*RI_background)**2 - Kx**2 - Ky**2)
    
    transfer_function = np.exp(1j*Kz*dz)

    # Forward propagation
    for z in range(RI_distribution.shape[2]):
        field_fft = np.fft.fft2(field)
        # plt.imshow(np.abs(field_fft))
        # plt.show()
        transfer_function = np.exp(1j*Kz*dz)
        phase = np.exp(1j*k0*(RI_distribution[..., z] - RI_background)*dz)
        if padding:
            phase = np.pad(phase, padding, 'edge') # no delay in the padded region
            
        field = np.fft.ifft2(field_fft * transfer_function) * phase
    
    if padding:
        field = field[padding:-1*padding, padding:-1*padding]
    # print(field.dtype)
    return field

 
def propagate(field: NDArray[np.complex128], 
              wavelength: float, 
              spatial_resolution: Tuple[float, float, float], 
              dist: float, 
              padding: Optional[int] = None, 
              direction: Optional[str] = 'forward', 
              bandlimited: Optional[bool] = False
    ) -> NDArray[np.complex128]:
    """
    Propagation through a homogenous medium. Base unit is meters.
    Use when input has to be updated.

    Args:
        field (NDArray[np.complex128]): 2d complex field on a plane
        wavelength (float): if not air, than wl =/ RI_background
        spatial_resolution (): _description_
        dist (float): distance bw parallel planes in meters

    Returns:
        NDArray[np.complex128]: field at parallel plane distance dist away 
    """
    if padding:
        field = np.pad(field, padding, 'edge') # edge make sense
        # symmteric would add diffraction pattern from outside FOV
        # zeros will make the aperture diffraction pattern dominates the diffraction patter after a certain distance
         
    k0 = 2 * np.pi / wavelength
    Nx, Ny = field.shape
    dx, dy = spatial_resolution[:2]
    
    # Spatial frequency grid
    kx = np.fft.fftfreq(Nx, dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, dy) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    Kz = 0j + k0**2 - Kx**2 - Ky**2
    if not np.all(Kz > 0):
        Kz[Kz < 0] = 0

    Kz = np.sqrt(Kz)
    
    field_fft = np.fft.fft2(field)
    if direction == 'backward':
        transfer_function = np.conj(np.exp(1j*Kz*dist))
    else:
        transfer_function = np.exp(1j*Kz*dist)
    
    if bandlimited:
        # max freq to prevent aliasing by the transfer function
        delU, delV = 1/(Nx*dx)*2*np.pi, 1/(Ny*dy)*2*np.pi
        uLimit = 1/(np.sqrt((2*delU*dist)**2 + 1)* wavelength)
        vLimit = 1/(np.sqrt((2*delV*dist)**2 + 1)* wavelength)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask = np.ones_like(transfer_function)
        mask[np.logical_or(np.abs(Kx) > int(uLimit), np.abs(Ky) >= int(vLimit))] = 0

        transfer_function = mask*transfer_function 
        
    field = np.fft.ifft2(field_fft * transfer_function)
    
    if padding:
        field = field[padding:-1*padding, padding:-1*padding]

    # print(field.dtype)
    return field
    

if __name__=='__main__':
    pass