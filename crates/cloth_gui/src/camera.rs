use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,   // ラジアン
    pub pitch: f32, // ラジアン
    pub fov_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::new(0.0, 0.0, 0.0),
            distance: 3.0,
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.35,
            fov_deg: 45.0,
            near: 0.01,
            far: 100.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye_position(&self) -> Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = -self.distance * self.pitch.cos() * self.yaw.cos();
        let z = self.distance * self.pitch.sin();
        self.target + Vec3::new(x, y, z)
    }

    pub fn view_projection(&self, aspect: f32) -> (Mat4, Vec3) {
        let eye = self.eye_position();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Z);
        let proj = Mat4::perspective_rh(self.fov_deg.to_radians(), aspect, self.near, self.far);
        (proj * view, eye)
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch -= dy * 0.008;
        // ピッチ角のクランプ (-89度 〜 89度)
        let limit = 89.0f32.to_radians();
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let eye = self.eye_position();
        let forward = (self.target - eye).normalize();
        let right = forward.cross(Vec3::Z).normalize();
        let up = right.cross(forward).normalize();

        let pan_speed = self.distance * 0.0015;
        self.target -= right * dx * pan_speed;
        self.target += up * dy * pan_speed;
    }

    pub fn zoom(&mut self, delta: f32) {
        let zoom_factor = (1.0 - delta * 0.1).clamp(0.1, 2.0);
        self.distance = (self.distance * zoom_factor).clamp(0.05, 50.0);
    }

    /// スクリーン座標 (x, y) からワールド空間のレイ (origin, direction) を算出する
    pub fn screen_to_ray(&self, screen_pos: (f32, f32), screen_size: (f32, f32)) -> (Vec3, Vec3) {
        let (width, height) = screen_size;
        if width <= 0.0 || height <= 0.0 {
            let eye = self.eye_position();
            return (eye, (self.target - eye).normalize());
        }
        let aspect = width / height;
        let (vp, eye) = self.view_projection(aspect);
        let inv_vp = vp.inverse();

        // スクリーン座標 -> NDC (-1.0 ..= 1.0)
        let ndc_x = (screen_pos.0 / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_pos.1 / height) * 2.0;

        let near_pt = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 0.0));
        let far_pt = inv_vp.project_point3(Vec3::new(ndc_x, ndc_y, 1.0));
        let dir = (far_pt - near_pt).normalize();
        (eye, dir)
    }

    /// ワールド3D座標をスクリーン2Dピクセル座標に射影する (成否, (sx, sy))
    pub fn world_to_screen(&self, world_pos: Vec3, screen_size: (f32, f32)) -> Option<(f32, f32)> {
        let (width, height) = screen_size;
        if width <= 0.0 || height <= 0.0 {
            return None;
        }
        let aspect = width / height;
        let (vp, _) = self.view_projection(aspect);
        let clip = vp * glam::Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);
        if clip.w <= 0.0 {
            return None; // カメラ後方
        }
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        let sx = (ndc_x + 1.0) * 0.5 * width;
        let sy = (1.0 - ndc_y) * 0.5 * height;
        Some((sx, sy))
    }
}
