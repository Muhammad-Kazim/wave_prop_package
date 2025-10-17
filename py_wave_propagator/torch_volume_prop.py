import torch
import torchvision
from typing import Optional, Tuple, List
from torch import nn, Tensor


def propagate_beam_vol(
    field: Tensor, 
    RI_distribution: Tensor, 
    RI_background: float, 
    wavelength: float, 
    spatial_resolution: Tuple[float, float, float], 
    padding: Optional[int] = None,
    pad_mode: str = 'edge'
    )-> Tensor:
    """
    Propagates a 2D complex wavefield using the beam propagationm method. Base units is meters.

    Args:
        field (Tensor): Input 2D complex wavefield in the (x, y, 0) plane
        RI_distribution (Tensor): 3D refractive index distribution (x, y, z)
        RI_background (float): background refractive index
        wavelength (float): wavelength of source in vacuum
        spatial_resolution (Tuple[float, float, float]): (dx, dy, dz). dz is the propagation step or slice thickness
        padding (Optional[int], optional): Number of pixels added to the field. Defaults to None.

    Returns:
        NDArray[np.complex64]: 2D complex wavefield at (x, y, -1)
    """
    PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode)
    if padding:
        field = PAD(field.real) + 1j*PAD(field.imag) # edge make sense
        # symmteric would add diffraction pattern from outside FOV
        # zeros will make the aperture diffraction pattern dominates the diffraction patter after a certain distance
        
    k0 = 2 * torch.pi / wavelength
    Nx, Ny = field.shape
    dx, dy, dz = spatial_resolution
    
    # Spatial frequency grid
    kx = torch.fft.fftfreq(Nx, dx) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, dy) * 2 * torch.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

    Kz = torch.sqrt(0j + (k0*RI_background)**2 - Kx**2 - Ky**2)
    
    transfer_function = torch.exp(1j*Kz*dz)

    # Forward propagation
    for z in range(RI_distribution.shape[2]):
        field_fft = torch.fft.fft2(field)
        # plt.imshow(np.abs(field_fft))
        # plt.show()
        transfer_function = torch.exp(1j*Kz*dz)
        phase = torch.exp(1j*k0*(RI_distribution[..., z] - RI_background)*dz)
        if padding:
            phase = PAD(phase.real) + 1j*PAD(phase.imag) # no delay in the padded region
            
        field = torch.fft.ifft2(field_fft * transfer_function) * phase
    
    if padding:
        return field[padding:-1*padding, padding:-1*padding]
    # print(field.dtype)
    return field

 
def propagate(field: Tensor, 
              wavelength: float, 
              spatial_resolution: Tuple[float, float, float], 
              dist: float, 
              padding: Optional[int] = None, 
              direction: Optional[str] = 'forward', 
              bandlimited: Optional[bool] = False,
              pad_mode: str = 'edge'
    ) -> Tensor:
    """
    Propagation through a homogenous medium. Base unit is meters.
    Use when input has to be updated.

    Args:
        field (Tensor): 2d complex field on a plane
        wavelength (float): if not air, than wl =/ RI_background
        spatial_resolution (): _description_
        dist (float): distance bw parallel planes in meters

    Returns:
        NDArray[np.complex128]: field at parallel plane distance dist away 
    """
    
    PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode) # edge make sense
    if padding:
        field = PAD(field.real) + 1j*PAD(field.imag) # does not handle complex properly
        # symmteric would add diffraction pattern from outside FOV
        # zeros will make the aperture diffraction pattern dominates the diffraction patter after a certain distance
         
    k0 = 2 * torch.pi / wavelength
    Nx, Ny = field.shape
    dx, dy = spatial_resolution[:2]
    
    # Spatial frequency grid
    kx = torch.fft.fftfreq(Nx, dx) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, dy) * 2 * torch.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    
    Kz = k0**2 - Kx**2 - Ky**2
    mask = torch.sigmoid(Kz/1e-12) # to remove negatives which remove evanascent waves
    Kz = torch.sqrt(Kz*mask)
    
    field_fft = torch.fft.fft2(field)
    if direction == 'backward':
        transfer_function = torch.conj(torch.exp(1j*Kz*dist))
    else:
        transfer_function = torch.exp(1j*Kz*dist)
    
    if bandlimited:
        # max freq to prevent aliasing by the transfer function
        delU, delV = 1/(Nx*dx)*2*torch.pi, 1/(Ny*dy)*2*torch.pi
        uLimit = 1/(torch.sqrt((2*delU*dist)**2 + 1)* wavelength)
        vLimit = 1/(torch.sqrt((2*delV*dist)**2 + 1)* wavelength)

        # limiting frequencies above uLimit and vLimit of the transfer function
        mask_u = 1 - torch.sigmoid((torch.abs(Kx) - (uLimit).int())/1e-12)
        mask_v = 1 - torch.sigmoid((torch.abs(Ky) - (vLimit).int())/1e-12)

        transfer_function = mask_u*mask_v*transfer_function 
        
    field = torch.fft.ifft2(field_fft * transfer_function)
    
    if padding:
        return field[padding:-1*padding, padding:-1*padding]

    # print(field.dtype)
    return field
    

if __name__=='__main__':
    pass