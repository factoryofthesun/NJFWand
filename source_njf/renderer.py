import torch
import numpy as np
import torch
import kaolin as kal

class Renderer:
    # from https://github.com/eladrich/latent-nerf
    def __init__(
        self,
        device,
        dim=(224, 224),
        interpolation_mode='bilinear',
        # Light Tensor (positive first): [ambient, right/left, front/back, top/bottom, ...]
        lights=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        import kaolin as kal
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()
        self.lights = lights.to(device)

    # @staticmethod
    # def get_camera_from_view_old(elev, azim, r=3.0, look_at_height=0.0):
    #     import kaolin as kal
    #     x = r * torch.sin(elev) * torch.sin(azim)
    #     y = r * torch.cos(elev)
    #     z = r * torch.sin(elev) * torch.cos(azim)

    #     pos = torch.tensor([x, y, z]).unsqueeze(0)
    #     look_at = torch.zeros_like(pos)
    #     look_at[:, 1] = look_at_height
    #     direction = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)

    #     camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    #     return camera_proj

    def get_camera_from_view(self, elev, azim, up=torch.tensor([0.0, 1.0, 0.0]), r=3.0, look_at_height=0.0,
                             return_cam=False, device=torch.device('cpu')):
        """
        Convert tensor elevation/azimuth values into camera projections

        Args:
            elev (torch.Tensor): elevation
            azim (torch.Tensor): azimuth
            r (float, optional): radius. Defaults to 3.0.

        Returns:
            Camera projection matrix (B x 4 x 3)
        """
        import kaolin as kal
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.sin(elev)
        z = r * torch.cos(elev) * torch.sin(azim)
        # print(elev,azim,x,y,z)
        B = elev.shape[0]

        if len(x.shape) == 0:
            pos = torch.tensor([x,y,z]).unsqueeze(0).to(device)
        else:
            pos = torch.stack([x, y, z], dim=1)
        # look_at = -pos
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height

        up = up.type(pos.dtype).unsqueeze(0).repeat(B, 1).to(device)
        # print(pos.shape, look_at.shape, up.shape)
        # from memory_debugging import CudaMemory
        # cm = CudaMemory()
        # cm.check_mem("before camera")
        # for i in range(1000000):
        #     a = 1+2
        #     b = a - 3 + 1
        camera_transform = kal.render.camera.generate_transformation_matrix(pos, look_at, up).to(device)

        # If return camera, then create camera object
        if return_cam:
            cams = kal.render.camera.Camera.from_args(
                    eye=pos,
                    at=look_at,
                    up=up,
                    fov= np.pi / 3, # Default based on __init__
                    width=self.dim[0], height=self.dim[1], device=self.device
                )
            return camera_transform, cams

        return camera_transform

    def render_texture(
        self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, up=torch.tensor([0.0, 1.0, 0.0]), radius=2,
        look_at_height=0.0, dims=None, white_background=False, vertexnormals=None,
        mod = False, specular = False,
        l_azim = [0., np.pi/2, np.pi, -np.pi/2, 0., 0.],
        l_elev = [0.] * 4 + [np.pi/2, -np.pi/2], amplitude = 1., sharpness = 3.,
    ):
        # uv face attr: B x F x 3 x 2
        # NOTE: Pytorch coordinates -1 to 1, yaxis from top to bottom -- circular filtering NOT supported
        import kaolin as kal
        dims = self.dim if dims is None else dims
        B = elev.shape[0]

        camera_transform, cam = self.get_camera_from_view(elev, azim, up=up, r=radius, look_at_height=look_at_height,
                                                     return_cam=True, device=self.device)

        # Vertices in camera coordinates (B x F x 3 x XYZ), vertices in image coordinates (B x F x 3 x 2),
        # face normals (B x F x 3)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        # UV: F x 3 x 2
        uv_face_attr = uv_face_attr.repeat(B, 1, 1, 1)

        if vertexnormals is None:
            fnormals = kal.ops.mesh.face_normals(verts[faces].unsqueeze(0), unit=True)
            vertexnormals = kal.ops.mesh.compute_vertex_normals(faces, fnormals.unsqueeze(2).repeat(1,1,3,1))

        normal_face_attr = vertexnormals[0, faces].repeat(B, 1, 1, 1).to(self.device)

        # TODO: sanity check -> single triangle render and check against a pixel loss
        # NOTE: We rasterize both UVs and normals per-pixel for correct shading
        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, [uv_face_attr, normal_face_attr])
        uv_features, normal_features = image_features

        # Apply mod function to UVs if set (don't grad this)
        if mod:
            with torch.no_grad():
                floorint = torch.floor(uv_features)
            uv_features = uv_features - floorint

        mask = (face_idx != -1)
        albedo = kal.render.mesh.texture_mapping(uv_features, texture_map.repeat(B, 1, 1, 1),
                                                         mode=self.interpolation_mode)
        albedo = torch.clamp(albedo * mask.unsqueeze(-1), 0., 1.)

        ### Add lighting
        # NOTE: Base lighting -- 6 lights from each primary direction
        l_azimuth = torch.tensor(l_azim, device=self.device).float()
        l_elevation = torch.tensor(l_elev, device=self.device).float()
        base_amplitude = torch.full((l_azimuth.shape[0], 3), amplitude, device=self.device).float()
        base_sharpness = torch.full((l_azimuth.shape[0],), sharpness, device=self.device).float()

        # If specular, then need to construct camera and generate pinhole rays + additional material params
        rays_d = base_spec = base_roughness = None
        if specular:
            base_spec = mask.unsqueeze(-1) * torch.tensor([1., 1., 1.], device=self.device)
            base_roughness = torch.full((B, *dims), 0.1, device=self.device)

            # Compute the rays
            rays_d = []
            for c in cam:
                rays_d.append(generate_pinhole_rays_dir(c, height=self.dim[0], width=self.dim[1]))
            # Rays must be toward the camera
            rays_d = -torch.cat(rays_d, dim=0)

        im_world_normal = torch.nn.functional.normalize(normal_features.detach(), p=2, dim=-1)
        img = add_lighting(mask, l_azimuth, l_elevation, base_amplitude, base_sharpness, im_world_normal,
           albedo, specular = specular, rays_d=rays_d, spec_albedo=base_spec, roughness=base_roughness)

        if white_background:
            img = img + (1 - mask.unsqueeze(-1).int())

        return img.permute(0, 3, 1, 2), mask.unsqueeze(1)

    def render_depth(
        self, verts, faces, elev=0, azim=0, up=torch.tensor([0.0, 1.0, 0.0]), radius=2,
        look_at_height=0.0, dims=None,
    ):
        # uv face attr: B x F x 3 x 2
        # NOTE: Pytorch coordinates -1 to 1, yaxis from top to bottom -- circular filtering NOT supported
        import kaolin as kal
        dims = self.dim if dims is None else dims
        B = elev.shape[0]

        camera_transform = self.get_camera_from_view(elev, azim, up=up, r=radius, look_at_height=look_at_height,
                                                     device=self.device)

        # Vertices in camera coordinates (B x F x 3 x XYZ), vertices in image coordinates (B x F x 3 x 2),
        # face normals (B x F x 3)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, [face_vertices_camera[:, :, :, [2]]])

        # NOTE: all z < 0; z closer to 0 is closer to cam
        # NOTE: We normalize [min, max] to [0, 1], so furthest pixel gets same value as background
        image_features = -1 * image_features[0]
        image_features -= torch.min(image_features)
        image_features /= torch.max(image_features)
        image_features = 1 - image_features # so further is darker

        # NOTE: Controlnet takes depth maps where closer = brighter and background = black
        albedo = image_features.repeat(1, 1, 1, 3)
        # albedo = torch.ones((image_features.shape[0], image_features.shape[1], image_features.shape[2], 3), device=image_features.device) * image_features
        mask = (face_idx != -1)

        # NOTE: This ensures that background is black
        img = torch.clamp(albedo * mask.unsqueeze(-1), 0., 1.)

        return img.permute(0, 3, 1, 2), mask.unsqueeze(1)

    def render_single_view(self, mesh, face_attributes, elev=0, azim=0, radius=2, look_at_height=0.0):
        import kaolin as kal
        dims = self.dim

        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height, device=self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        mask = (face_idx > -1).float()[..., None]

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)

    def render_mesh(self, vertices, faces, colors, elev=torch.zeros([1]), azim=torch.zeros([1]), radius=2, look_at_height=0.0, dims=None,
                    white_background=True, up=torch.tensor([0.0, 1.0, 0.0]), vertexnormals=None,
                    l_azim = [0., np.pi/2, np.pi, -np.pi/2, 0., 0.],
                    l_elev = [0.] * 4 + [np.pi/2, -np.pi/2], amplitude = 1., sharpness = 3., specular = False):
        import kaolin as kal
        dims = self.dim if dims is None else dims

        import kaolin.ops.mesh
        B = elev.shape[0]
        face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
                                colors.unsqueeze(0),
                                faces.long()
                            )
        face_attributes = face_attributes.repeat(B, 1, 1, 1).to(self.device)

        # Need normals for shading
        if vertexnormals is None:
            fnormals = kal.ops.mesh.face_normals(vertices[faces].unsqueeze(0), unit=True)
            vertexnormals = kal.ops.mesh.compute_vertex_normals(faces.long(), fnormals.unsqueeze(2).repeat(1,1,3,1))
        normal_face_attr = vertexnormals[0, faces].repeat(B, 1, 1, 1).to(self.device)

        camera_transform = self.get_camera_from_view(elev, azim, r=radius,
                                                look_at_height=look_at_height, device=self.device,
                                                up=up)

        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
           vertices, faces, self.camera_projection, camera_transform=camera_transform)

        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, [face_attributes, normal_face_attr])
        albedo, normal_features = image_features

        mask = (face_idx != -1)
        albedo = torch.clamp(albedo * mask.unsqueeze(-1), 0., 1.)

        ### Add lighting
        # NOTE: Base lighting -- 6 lights from each primary direction
        l_azimuth = torch.tensor(l_azim, device=self.device).float()
        l_elevation = torch.tensor(l_elev, device=self.device).float()
        base_amplitude = torch.full((l_azimuth.shape[0], 3), amplitude, device=self.device).float()
        base_sharpness = torch.full((l_azimuth.shape[0],), sharpness, device=self.device).float()

        # If specular, then need to construct camera and generate pinhole rays + additional material params
        rays_d = base_spec = base_roughness = None
        if specular:
            base_spec = mask.unsqueeze(-1) * torch.tensor([1., 1., 1.], device=self.device)
            base_roughness = torch.full((B, *dims), 0.1, device=self.device)

            # Compute the rays
            rays_d = []
            for c in cam:
                rays_d.append(generate_pinhole_rays_dir(c, height=self.dim[0], width=self.dim[1]))
            # Rays must be toward the camera
            rays_d = -torch.cat(rays_d, dim=0)

        im_world_normal = torch.nn.functional.normalize(normal_features.detach(), p=2, dim=-1)
        img = add_lighting(mask, l_azimuth, l_elevation, base_amplitude, base_sharpness, im_world_normal,
           albedo, specular = specular, rays_d=rays_d, spec_albedo=base_spec, roughness=base_roughness)

        if white_background:
            img = img + (1 - mask.unsqueeze(-1).int())

        return img.permute(0, 3, 1, 2), mask.unsqueeze(1)

#----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
#----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):
    return np.array([[n/x,    0,            0,              0],
                     [  0, n/-x,            0,              0],
                     [  0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)],
                     [  0,    0,           -1,              0]]).astype(np.float32)

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0],
                     [0,  c, s, 0],
                     [0, -s, c, 0],
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0],
                     [ 0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def scale(s):
    return np.array([[ s, 0, 0, 0],
                     [ 0, s, 0, 0],
                     [ 0, 0, s, 0],
                     [ 0, 0, 0, 1]]).astype(np.float32)

def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate = np.array([[1, 0, 0, -eye[0]],
                          [0, 1, 0, -eye[1]],
                          [0, 0, 1, -eye[2]],
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[u[0], u[1], u[2], 0],
                        [v[0], v[1], v[2], 0],
                        [w[0], w[1], w[2], 0],
                        [0, 0, 0, 1]]).astype(np.float32)
    return np.matmul(rotate, translate)

def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m

#----------------------------------------------------------------------------
# Render Helpers
#----------------------------------------------------------------------------

def generate_pinhole_rays_dir(camera, height, width, device='cuda'):
    """Generate centered grid.

    This is a utility function for specular reflectance with spherical gaussian.
    """
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    pixel_x = pixel_x + 0.5  # scale and add bias to pixel center
    pixel_y = pixel_y + 0.5  # scale and add bias to pixel center

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    # Convert to NDC
    pixel_x = 2 * (pixel_x / width) - 1.0
    pixel_y = 2 * (pixel_y / height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1).float()

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

    return ray_dir[0].reshape(1, height, width, 3)

# Given albedo and lighting parameters, add lighting to the render
def add_lighting(hard_mask, l_azimuth, l_elevation, amplitude, sharpness, im_world_normal,
           albedo, specular = False, rays_d=None, spec_albedo=None, roughness=None):
    """Render diffuse and specular components.

    Use spherical gaussian fitted approximation for the diffuse component"""
    batch_size = albedo.shape[0]

    # Add lighting components broadcasted over batch dimension
    directions = torch.stack(kal.ops.coords.spherical2cartesian(l_azimuth, l_elevation), dim=-1)
    img = torch.zeros(im_world_normal.shape, device='cuda', dtype=torch.float32)

    # NOTE: May need to repeat amp/dir/sharp over batch size
    # Render diffuse component
    diffuse_effect = kal.render.lighting.sg_diffuse_inner_product(
        amplitude,
        directions,
        sharpness,
        im_world_normal[hard_mask],
        albedo[hard_mask]
    )

    if specular:
        assert rays_d is not None
        assert spec_albedo is not None
        assert roughness is not None

        # Render specular component
        specular_effect = kal.render.lighting.sg_warp_specular_term(
            amplitude,
            directions,
            sharpness,
            im_world_normal[hard_mask],
            roughness[hard_mask].float(),
            rays_d[hard_mask],
            spec_albedo[hard_mask].float()
        )
        img[hard_mask] = diffuse_effect + specular_effect
    else:
        img[hard_mask] = diffuse_effect

    # HDR: Rescale to [0, 1]
    if torch.max(img) > 1:
        img = img / torch.max(img)

    return img

def transform_pos(mtx, pos):
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).float().to(pos.device)], axis=1)
    return torch.matmul(posw, mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    import nvdiffrast.torch as dr
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.
    return color
