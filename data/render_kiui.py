import os
import cv2
import tqdm
import argparse
import numpy as np
import json
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

GUI_AVAILABLE = True
try:
    import dearpygui.dearpygui as dpg
except Exception as e:
    # print('[WARN] cannot import dearpygui, assume running with --wogui')
    GUI_AVAILABLE = False

from kiui.mesh import Mesh
from kiui.cam import OrbitCamera
from kiui.op import safe_normalize

X270 = ['dreamfusion','grm','grm_sv3d','latent-nerf','LucidDreamer_glb','magic3d','magic123','mvdream','OpenLRM','sjc','SyncDreamer','sz123','TriplaneGaussian_glb','triposr','z123']
X90 = ['InstantMesh',]
Y180 = ['crm','spard_glb','Wonder3D',]
Y90 = ['grm_sv3d',]
Y270 = ['dreamfusion','InstantMesh','latent-nerf','LucidDreamer_glb','magic3d','magic123','mvdream','SyncDreamer','sz123','TriplaneGaussian_glb','triposr','z123']

def compute_face_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]  # [F, 3]
    v1 = vertices[faces[:, 1]]  # [F, 3]
    v2 = vertices[faces[:, 2]]  # [F, 3]
    edge1 = v1 - v0  # [F, 3]
    edge2 = v2 - v0  # [F, 3]
    face_normals = torch.cross(edge1, edge2, dim=-1)  # [F, 3]
    face_normals = F.normalize(face_normals, p=2, dim=-1)  # [F, 3]
    return face_normals.to(torch.int32)

def rotate_mesh(mesh, axis='y', angle_degrees=90):
    angle = np.radians(angle_degrees)
    vertices = mesh.v.clone()
    
    if axis == 'x':
        rot = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rot = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
    rot = torch.tensor(rot, device=vertices.device, dtype=vertices.dtype)
    mesh.v = vertices @ rot.T

    if hasattr(mesh, 'vn'):
        mesh.vn = mesh.vn @ rot.T
        mesh.vn = F.normalize(mesh.vn, p=2, dim=-1)
    
    if hasattr(mesh, 'f'):
        mesh.fn = compute_face_normals(mesh.v, mesh.f)
    
    if hasattr(mesh, 'tangent'):
        mesh.tangent = mesh.tangent @ rot.T
        mesh.tangent = F.normalize(mesh.tangent, p=2, dim=-1)
    if hasattr(mesh, 'bitangent'):
        mesh.bitangent = mesh.bitangent @ rot.T
        mesh.bitangent = F.normalize(mesh.bitangent, p=2, dim=-1)

    return mesh

class GUI:
    def __init__(self, opt):
        self.opt = opt
        self.W = opt.W
        self.H = opt.H
        if not GUI_AVAILABLE and not opt.wogui:
            print(f'[WARN] cannot import dearpygui, assume running with --wogui')
        self.wogui = not GUI_AVAILABLE or opt.wogui # disable gui and run in cmd
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        ### white bg
        self.bg_color = torch.ones(3, dtype=torch.float32).cuda()
        ### black bg
        # self.bg_color = torch.zeros(3, dtype=torch.float32).cuda()

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.light_dir = np.array([0, 0])
        self.ambient_ratio = 0.5
        # self.ambient_ratio = 0.0

        # auto-rotate
        self.auto_rotate_cam = False
        self.auto_rotate_light = False
        
        # load mesh
        try:

            self.mesh = Mesh.load(opt.mesh, front_dir=opt.front_dir)
            method = os.path.basename(os.path.dirname(opt.mesh))
            if method in X90:
                self.mesh = rotate_mesh(self.mesh, axis='x', angle_degrees=90)
            elif method in X270:
                self.mesh = rotate_mesh(self.mesh, axis='x', angle_degrees=270)
            if method in Y180:
                self.mesh = rotate_mesh(self.mesh, axis='y', angle_degrees=90)
            elif method in Y90:
                self.mesh = rotate_mesh(self.mesh, axis='y', angle_degrees=90)
            elif method in Y270:
                self.mesh = rotate_mesh(self.mesh, axis='y', angle_degrees=270)

        except Exception as e:
            self.mesh = None
            with open('e.log','a') as f:
                f.write('*'*10)
                f.write('\n')
                f.write(opt.mesh)
                f.write('\n')
                f.write(str(e))
                f.write('\n')
            return
            
        # render_mode
        self.render_modes = ['depth', 'normal', 'shading']
        if self.mesh.albedo is not None or self.mesh.vc is not None:
            self.render_modes.extend(['albedo', 'lambertian'])

        # display wireframe
        self.show_wire = False
        self.wire_width = 0.01
        self.wire_color = np.array([0, 0, 0], dtype=np.float32)

        # load pbr if enabled
        if self.opt.pbr:
            import envlight
            if self.opt.envmap is None:
                hdr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/lights/mud_road_puresky_1k.hdr')
            else:
                hdr_path = self.opt.envmap
            self.light = envlight.EnvLight(hdr_path, scale=2, device='cuda')
            self.FG_LUT = torch.from_numpy(np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/lights/bsdf_256_256.bin"), dtype=np.float32).reshape(1, 256, 256, 2)).cuda()

            self.metallic_factor = 1
            self.roughness_factor = 1

            self.render_modes.append('pbr')
            
        if opt.mode in self.render_modes:
            self.mode = opt.mode
        else:
            print(f'[WARN] mode {opt.mode} not supported, fallback to render shading')
            self.mode = 'shading' # fallback
        
        self.glctx = dr.RasterizeCudaContext()#dr.RasterizeGLContext()

        if not self.wogui:
            dpg.create_context()
            self.register_dpg()
            self.step()
        

    def __del__(self):
        if not self.wogui:
            dpg.destroy_context()
    
    def step(self):

        if not self.need_update:
            return
    
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        # do MVP for vertices
        pose = torch.from_numpy(self.cam.pose.astype(np.float32)).cuda()
        proj = torch.from_numpy(self.cam.perspective.astype(np.float32)).cuda()
        
        v_cam = torch.matmul(F.pad(self.mesh.v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        H, W = int(self.opt.ssaa * self.H), int(self.opt.ssaa * self.W)
        rast, rast_db = dr.rasterize(self.glctx, v_clip, self.mesh.f, (H, W))

        alpha = (rast[..., 3:] > 0).float()
        alpha = dr.antialias(alpha, rast, v_clip, self.mesh.f).squeeze(0).clamp(0, 1) # [H, W, 3]
        
        if self.mode == 'depth':
            depth, _ = dr.interpolate(-v_cam[..., [2]], rast, self.mesh.f) # [1, H, W, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
            buffer = depth.squeeze(0).detach().cpu().numpy().repeat(3, -1) # [H, W, 3]
        elif self.mode == 'normal':
            normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
            normal = safe_normalize(normal)
            normal_image = (normal[0] + 1) / 2
            normal_image = torch.where(rast[..., 3:] > 0, normal_image, torch.tensor(1.0).to(normal_image.device)) # remove background
            buffer = normal_image[0].detach().cpu().numpy()
        elif self.mode == 'shading':
            self.bg_color = torch.zeros(3, dtype=torch.float32).cuda()
            self.mesh.vc = torch.ones_like(self.mesh.v)
            albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0.0).to(albedo.device)) # remove background
            normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
            normal = safe_normalize(normal)
            light_d = np.deg2rad(self.light_dir)
            light_d = np.array([
                np.cos(light_d[0]) * np.sin(light_d[1]),
                -np.sin(light_d[0]),
                np.cos(light_d[0]) * np.cos(light_d[1]),
            ], dtype=np.float32)
            light_d = torch.from_numpy(light_d).to(albedo.device)
            lambertian = self.ambient_ratio + (1 - self.ambient_ratio) * (normal @ light_d).float().clamp(min=0)
            albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
            buffer = albedo[0].detach().cpu().numpy()
        else:
            # use vertex color if exists
            if self.mesh.vc is not None:
                albedo, _ = dr.interpolate(self.mesh.vc.unsqueeze(0).contiguous(), rast, self.mesh.f)
            # use texture image
            else: # assert mesh.albedo is not None
                texc, _ = dr.interpolate(self.mesh.vt.unsqueeze(0).contiguous(), rast, self.mesh.ft)
                albedo = dr.texture(self.mesh.albedo.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]

            albedo = torch.where(rast[..., 3:] > 0, albedo, torch.tensor(0.0).to(albedo.device)) # remove background
            # albedo = dr.antialias(albedo, rast, v_clip, self.mesh.f).clamp(0, 1) # [1, H, W, 3]

            if self.mode == 'albedo':
                albedo = albedo * alpha + self.bg_color * (1 - alpha)
                buffer = albedo[0].detach().cpu().numpy()
            else:
                normal, _ = dr.interpolate(self.mesh.vn.unsqueeze(0).contiguous(), rast, self.mesh.fn)
                normal = safe_normalize(normal)
                
                if self.mode == 'lambertian':
                    light_d = np.deg2rad(self.light_dir)
                    light_d = np.array([
                        np.cos(light_d[0]) * np.sin(light_d[1]),
                        -np.sin(light_d[0]),
                        np.cos(light_d[0]) * np.cos(light_d[1]),
                    ], dtype=np.float32)
                    light_d = torch.from_numpy(light_d).to(albedo.device)
                    lambertian = self.ambient_ratio + (1 - self.ambient_ratio) * (normal @ light_d).float().clamp(min=0)
                    albedo = (albedo * lambertian.unsqueeze(-1)) * alpha + self.bg_color * (1 - alpha)
                    buffer = albedo[0].detach().cpu().numpy()
                    
                elif self.mode == 'pbr':
                    if self.mesh.metallicRoughness is not None:
                        metallicRoughness = dr.texture(self.mesh.metallicRoughness.unsqueeze(0), texc, filter_mode='linear') # [1, H, W, 3]
                        metallic = metallicRoughness[..., 2:3] * self.metallic_factor
                        roughness = metallicRoughness[..., 1:2] * self.roughness_factor
                    else:
                        metallic = torch.ones_like(albedo[..., :1]) * self.metallic_factor
                        roughness = torch.ones_like(albedo[..., :1]) * self.roughness_factor

                    xyzs, _ = dr.interpolate(self.mesh.v.unsqueeze(0), rast, self.mesh.f) # [1, H, W, 3]
                    viewdir = safe_normalize(xyzs - pose[:3, 3])

                    n_dot_v = (normal * viewdir).sum(-1, keepdim=True) # [1, H, W, 1]
                    reflective = n_dot_v * normal * 2 - viewdir

                    diffuse_albedo = (1 - metallic) * albedo

                    fg_uv = torch.cat([n_dot_v, roughness], -1).clamp(0, 1) # [H, W, 2]
                    fg = dr.texture(
                        self.FG_LUT,
                        fg_uv.reshape(1, -1, 1, 2).contiguous(),
                        filter_mode="linear",
                        boundary_mode="clamp",
                    ).reshape(1, H, W, 2)
                    F0 = (1 - metallic) * 0.04 + metallic * albedo
                    specular_albedo = F0 * fg[..., 0:1] + fg[..., 1:2]

                    diffuse_light = self.light(normal)
                    specular_light = self.light(reflective, roughness)

                    color = diffuse_albedo * diffuse_light + specular_albedo * specular_light # [H, W, 3]
                    color = color * alpha + self.bg_color * (1 - alpha)

                    buffer = color[0].detach().cpu().numpy()

        if self.show_wire:
            u = rast[..., 0] # [1, h, w]
            v = rast[..., 1] # [1, h, w]
            w = 1 - u - v
            mask = rast[..., 2]
            near_edge = (((w < self.wire_width) | (u < self.wire_width) | (v < self.wire_width)) & (mask > 0))[0].detach().cpu().numpy() # [h, w]
            buffer[near_edge] = self.wire_color
        
        # ssaa rescale
        if H != self.H or W != self.W:
            buffer = cv2.resize(buffer, (self.H, self.W), interpolation=cv2.INTER_AREA)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.render_buffer = buffer
        self.need_update = False

        if self.auto_rotate_cam:
            self.cam.orbit(5, 0)
            self.need_update = True
        
        if self.auto_rotate_light:
            self.light_dir[1] += 3
            self.need_update = True
        
        if not self.wogui:
            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=200, collapsed=True):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)              

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(self.render_modes, label='mode', default_value=self.mode, tag="_mode_combo", callback=callback_change_mode)

                # show wireframe
                def callback_toggle_wireframe(sender, app_data):
                    self.show_wire = not self.show_wire
                    dpg.set_value("_checkbox_wire", self.show_wire)
                    self.need_update = True

                dpg.add_checkbox(label="wireframe", tag="_checkbox_wire", default_value=self.show_wire, callback=callback_toggle_wireframe)

                # wireframe width
                def callback_set_wire_width(sender, app_data):
                    self.wire_width = app_data
                    self.need_update = True

                dpg.add_slider_float(label="wireframe width", min_value=0, max_value=1.0, format="%.5f", default_value=self.wire_width, callback=callback_set_wire_width)

                # wire_color picker
                def callback_change_wire_color(sender, app_data):
                    self.wire_color = np.array(app_data[:3], dtype=np.float32)
                    self.need_update = True
                
                dpg.add_color_edit((0, 0, 0), label="Wireframe Color", width=200, no_alpha=True, callback=callback_change_wire_color)


                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32).cuda() # only need RGB in [0, 1]
                    self.need_update = True
                
                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(label="FoVY", min_value=1, max_value=120, format="%d deg", default_value=np.rad2deg(self.cam.fovy), callback=callback_set_fovy)

                # light dir
                def callback_set_light_dir(sender, app_data, user_data):
                    self.light_dir[user_data] = app_data
                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Plane Light Direction:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="elevation", min_value=-90, max_value=90, format="%.2f", default_value=self.light_dir[0], callback=callback_set_light_dir, user_data=0)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="azimuth", min_value=0, max_value=360, format="%.2f", default_value=self.light_dir[1], callback=callback_set_light_dir, user_data=1)

                # ambient ratio
                def callback_set_abm_ratio(sender, app_data):
                    self.ambient_ratio = app_data
                    self.need_update = True

                dpg.add_slider_float(label="ambient", min_value=0, max_value=1.0, format="%.5f", default_value=self.ambient_ratio, callback=callback_set_abm_ratio)

                # pbr
                if self.opt.pbr:
                    # metallic
                    def callback_set_metallic(sender, app_data):
                        self.metallic_factor = app_data
                        self.need_update = True

                    dpg.add_slider_float(label="metallic", min_value=0, max_value=1.0, format="%.5f", default_value=self.metallic_factor, callback=callback_set_metallic)
                
                    # roughness
                    def callback_set_roughness(sender, app_data):
                        self.roughness_factor = app_data
                        self.need_update = True

                    dpg.add_slider_float(label="roughness", min_value=0, max_value=1.0, format="%.5f", default_value=self.roughness_factor, callback=callback_set_roughness)

        ### register IO handlers

        # camera mouse controller
        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        # press spacebar to toggle rendering mode
        def callback_space_toggle_mode(sender, app_data):
            self.mode = self.render_modes[(self.render_modes.index(self.mode) + 1) % len(self.render_modes)]
            dpg.set_value("_mode_combo", self.mode)
            self.need_update = True
        
        # press P to toggle auto-rotate camera
        def callback_toggle_auto_rotate_cam(sender, app_data):
            self.auto_rotate_cam = not self.auto_rotate_cam
            self.need_update = True
        
        # press L to toggle auto-rotate light
        def callback_toggle_auto_rotate_light(sender, app_data):
            self.auto_rotate_light = not self.auto_rotate_light
            self.need_update = True

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Right, callback=callback_camera_drag_pan)

            dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=callback_space_toggle_mode)
            dpg.add_key_press_handler(dpg.mvKey_P, callback=callback_toggle_auto_rotate_cam)
            dpg.add_key_press_handler(dpg.mvKey_L, callback=callback_toggle_auto_rotate_light)
            dpg.add_key_press_handler(dpg.mvKey_W, callback=callback_toggle_wireframe)

        
        dpg.create_viewport(title='mesh viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        assert not self.wogui
        while dpg.is_dearpygui_running():
            self.step()
            dpg.render_dearpygui_frame()

def kiui_render(opt):

    gui = GUI(opt)
    if gui.mesh is None:
        return
    save_image = opt.save_image if opt.save_image else 'temp' if opt.save_video else None
    if save_image is not None:
        images = []
        os.makedirs(save_image, exist_ok=True)
        # render from fixed views and save all images
        elevation = [opt.elevation,]
        azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
        for ele in elevation:
            for num,azi in enumerate(azimuth):
                if opt.object:
                    gui.cam.from_angle(ele, azi)
                    gui.light_dir = np.array([ele,azi])
                elif opt.light:
                    gui.cam.from_angle(0, 0)
                    gui.light_dir = np.array([ele, azi])
                gui.need_update = True
                gui.step()
                if not opt.wogui:
                    dpg.render_dearpygui_frame()
                image = (gui.render_buffer * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_image, f'{str(num).zfill(3)}.png'), image)
                if opt.save_video is not None:
                    images.append(image)
        if opt.save_video is not None:
            # import imageio
            # images = np.stack(images, axis=0)
            # # ~4 seconds, 120 frames at 30 fps
            # imageio.mimwrite(opt.save_video+'.gif', images, duration=2, quality=8, macro_block_size=1)
            os.system(f"convert -delay 10 -loop 0 {save_image}/*.png {opt.save_video}.gif")
    else:
        gui.render()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, help="json file of mesh path (obj, glb)")
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--object',action='store_true')
    parser.add_argument('--light',action='store_true')
    parser.add_argument('--direction',type=str,choices=['up','down','left','right'])
    parser.add_argument('--pbr', action='store_true', help="enable PBR material")
    parser.add_argument('--envmap', type=str, default=None, help="hdr env map path for pbr")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='shading', type=str, choices=['lambertian', 'albedo', 'normal', 'depth', 'pbr', 'shading'], help="rendering mode")
    parser.add_argument('--W', type=int, default=512, help="GUI width")
    parser.add_argument('--H', type=int, default=512, help="GUI height")
    parser.add_argument('--ssaa', type=float, default=1, help="super-sampling anti-aliasing ratio")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument("--wogui", action='store_true', help="disable all dpg GUI")
    parser.add_argument('--save_image', type=str, default=None, help="path to save example rendered images")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=40, help="number of images to render from different azimuths")
    parser.add_argument('--save_video', type=str, default=None, help="path to save rendered video")
    opt = parser.parse_args()

    with open(opt.mesh,'r') as f:
        # data = f.readlines()
        data = json.load(f)
        data = list(data.keys())
        data.sort()

    step = len(data)/opt.split
    data = data[int(step*opt.index):int(step*(opt.index+1))]

    save_path_root = opt.save_image if opt.save_image else opt.save_video if opt.save_video else None
    if save_path_root and data:
        os.makedirs(save_path_root,exist_ok=True)
        os.makedirs(os.path.join(save_path_root,opt.mode),exist_ok=True)

    for index, i in enumerate(data):
        path = os.path.abspath(i.replace('\n',''))
        opt.mesh = path
        sub_path = os.path.basename(os.path.dirname(os.path.dirname(path)))
        obj_name = os.path.splitext(os.path.basename(path))[0]
        if opt.save_image is not None:
            os.makedirs(os.path.join(save_path_root,opt.mode,sub_path),exist_ok=True)
            opt.save_image = os.path.join(save_path_root,opt.mode,sub_path)
        elif opt.save_video is not None:
            os.makedirs(os.path.join(save_path_root,opt.mode,sub_path),exist_ok=True)
            opt.save_video = os.path.join(save_path_root,opt.mode,sub_path,obj_name)
        kiui_render(opt)
        # break
        