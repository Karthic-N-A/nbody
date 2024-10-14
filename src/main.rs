use error_iter::ErrorIter as _;
use glam::f32::Vec2;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;
use rand::distributions::{Distribution,Uniform};


const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const N: usize = 50_000;
const GRAV: f32 = 1.; // gravitation constant
const THETA: f32 = 0.6; // Parameter affected both quality and speed. Too high, quality is low, too low fps is low
const SOFTENING: f32 = 10.; // softening parameter based on wikipedia article on nbody
const M: f32 = 1e4; // mass of central body

#[derive(Clone,Copy)]
struct Particle {
    mass: f32,
    r: Vec2,
    v: Vec2,
    field: Vec2,
    field_prev: Vec2,
    first_iter: bool 
}

impl Default for Particle{
    fn default() -> Self {
        Self{
            mass: 1.,
            r: Vec2::ZERO,
            v: Vec2::ZERO,
            field: Vec2::ZERO,
            field_prev: Vec2::ZERO,
            first_iter: true,
        }
    }
}

// An enum for quad tree, either holds a final node with Some(index of particle) or None
// or Branch, which is 4 quadrants within it
enum QuadTree{
    Leaf(Option<usize>),
    Branch([Box<QuadNode>; 4]),
}

// Metadata associated with Quadtree, such as position, width, height, mass contained within the
// node, total_mass*center_of_mass and quadtree itself
struct QuadNode {
    top_left: Vec2,
    width: f32,
    height: f32,
    mass: f32,
    center_of_mass_sum: Vec2,
    qt: QuadTree,
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut input = WinitInputHelper::new();
    let mut rng = rand::thread_rng();
    let uradius = Uniform::from(20.0..100.);
    let utheta = Uniform::from(-std::f32::consts::PI..std::f32::consts::PI);
    let mut now = std::time::Instant::now();
    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Barnes Hut")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };
    // Boilerplate for pixels and winit setup

    let mut particles : Vec<Particle> = Vec::new();
    particles.push(Particle{mass:M, r:Vec2{x: (WIDTH/2) as f32, y:(HEIGHT/2) as f32}, ..Default::default()});
    // Make N random particles spread around center
    for _ in 1..=N{
        let r = uradius.sample(&mut rng);
        let t = utheta.sample(&mut rng);
        let v = (GRAV*M/r).sqrt();
        particles.push(Particle {
            mass: 1.,
            r: Vec2 {
                x: (WIDTH / 2) as f32 + r*t.cos(),
                y: (HEIGHT / 2) as f32 + r*t.sin()
            },
            v: Vec2{
                x: -v*t.sin(),
                y: v*t.cos(),
            },
            // r is perpendicular to v
            ..Default::default()
        });
    }


    // Used for deltatime later on
    let dt = 1./60.;

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH, HEIGHT, surface_texture)?
    };

    let res = event_loop.run(move |event, elwt| {
        // Draw the current frame
        if let Event::WindowEvent {
            event: WindowEvent::RedrawRequested,
            ..
        } = event
        {
            let frame = pixels.frame_mut();
            frame.fill(0);
            for p in &particles {
                let x = p.r.x as u32;
                let y = p.r.y as u32;
                // only render if within screen
                if 0 < x && x < WIDTH && 0 < y && y < HEIGHT {
                    let i = (WIDTH * y + x) as usize;
                    // transition from blue to red based on magnitude of velocity
                    let d:f32 = match p.v.length(){
                        0.0..1. => 1.,
                        e => 1./e,
                    };
                    let rgba = [(255.*(1.-d)) as u8 , 40, (255.*(d)) as u8, 0xff];
                    frame[4 * i..(4 * i) + 4].copy_from_slice(&rgba);
                }
            }

            if let Err(err) = pixels.render() {
                log_error("pixels.render", err);
                elwt.exit();
                return;
            }
        }

        // Handle input events
        if input.update(&event) {
            if input.key_held(KeyCode::Escape) || input.close_requested() {
                elwt.exit();
                return;
            }
            if let Some(size) = input.window_resized() {
                if let Err(err) = pixels.resize_surface(size.width, size.height) {
                    log_error("pixels.resize_surface", err);
                    elwt.exit();
                    return;
                }
            }

            // Initialise the root node
            let mut root = QuadNode {
                top_left: Vec2::ZERO,
                width: WIDTH as f32,
                height: HEIGHT as f32,
                center_of_mass_sum: Vec2::ZERO,
                mass: 0.,
                qt: QuadTree::Leaf(None),
            };

            // build tree
            for i in 0..particles.len(){
                // If particle is not within the screen, respawm it as a new particle somewhere inside
                if !(0. < particles[i].r.x && particles[i].r.x < WIDTH as f32 && 0. < particles[i].r.y && particles[i].r.y < HEIGHT as f32){
                    let r = uradius.sample(&mut rng);
                    let t = utheta.sample(&mut rng);
                    particles[i].r = Vec2 {x: (WIDTH / 2) as f32 + r*t.cos(), y: (HEIGHT/2) as f32 + r*t.sin()};
                    particles[i].v = Vec2{x: -r/100.*t.sin(), y: r/100.*t.cos()};
                    particles[i].field = Vec2::ZERO;
                    particles[i].first_iter = true;
                }
                // Try adding the particle to root
                put(&mut root, i, &particles);
            }

            for i in 0..particles.len(){
                calculate_field(&root, i, &mut particles);
            }
            for i in 0..particles.len(){
                // using particles[i].v = particles[i].field*dt returns an error, telling to use a
                // local variable, so instead handle each field
                if !particles[i].first_iter{
                    particles[i].v.x += dt/2. * (particles[i].field_prev.x + particles[i].field.x);
                    particles[i].v.y += dt/2. * (particles[i].field_prev.y + particles[i].field.y);
                }
                else{
                    particles[i].first_iter = false;
                }
                // TODO: Decide if it is efficient to have a bool check everytime despite the
                // fact only checking once is needed. Otherwise make a separate first run outside the loop just
                // for this(but then code duplication)
                particles[i].field_prev = particles[i].field;
                particles[i].field = Vec2::ZERO;
                particles[i].r.x += particles[i].v.x * dt + particles[i].field.x/2. * dt * dt;
                particles[i].r.y += particles[i].v.y * dt + particles[i].field.y/2. * dt * dt;
            }
            println!("{}", now.elapsed().as_secs_f32().recip());
            now = std::time::Instant::now();


            window.request_redraw();
        }
    });
    res.map_err(|e| Error::UserDefined(Box::new(e)))
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("Caused by: {source}");
    }
}

fn put(node: &mut QuadNode, i: usize, particles: &Vec<Particle>) {
    let p = particles[i];
    match &mut node.qt {
        QuadTree::Leaf(particle) => {
            node.center_of_mass_sum += p.mass*p.r;
            node.mass += p.mass;
            match particle.take() {
                // If the leaf is empty, simply add particle
                None => {
                    node.qt = QuadTree::Leaf(Some(i));
                },
                // If the leaf is occupied, split it into quadrants and to individual quad
                // The quadtree associated with the node becomes a branch
                Some(u) => {
                    node.qt = QuadTree::Branch([
                        Box::new(QuadNode {top_left: Vec2 {x: node.top_left.x + node.width / 2., y: node.top_left.y                    },width: node.width / 2.,height: node.height / 2.,center_of_mass_sum:Vec2::ZERO, mass: 0.,qt: QuadTree::Leaf(None)}),
                        Box::new(QuadNode {top_left: Vec2 {x: node.top_left.x,                   y: node.top_left.y                    },width: node.width / 2.,height: node.height / 2.,center_of_mass_sum:Vec2::ZERO,mass: 0.,qt: QuadTree::Leaf(None)}),
                        Box::new(QuadNode {top_left: Vec2 {x: node.top_left.x,                   y: node.top_left.y + node.height / 2. },width: node.width / 2.,height: node.height / 2.,center_of_mass_sum:Vec2::ZERO,mass: 0.,qt: QuadTree::Leaf(None),}),
                        Box::new(QuadNode {top_left: Vec2 {x: node.top_left.x + node.width / 2., y: node.top_left.y + node.height / 2. },width: node.width / 2.,center_of_mass_sum:Vec2::ZERO,height: node.height / 2.,mass: 0.,qt: QuadTree::Leaf(None)}),
                    ]);
                    put(node, u, particles);
                    put(node, i, particles);
                }
            }
        },
        // check which leaf in branch has the coordinates required to fit in the particle, and try
        // putting in it
        QuadTree::Branch(branch) => {
            if      branch[0].top_left.x < p.r.x && p.r.x < branch[0].top_left.x + branch[0].width && branch[0].top_left.y < p.r.y && p.r.y < branch[0].top_left.y + branch[0].height  {put(&mut branch[0], i, particles);}
            else if branch[1].top_left.x < p.r.x && p.r.x < branch[1].top_left.x + branch[1].width && branch[1].top_left.y < p.r.y && p.r.y < branch[1].top_left.y + branch[1].height  {put(&mut branch[1], i, particles);}
            else if branch[2].top_left.x < p.r.x && p.r.x < branch[2].top_left.x + branch[2].width && branch[2].top_left.y < p.r.y && p.r.y < branch[2].top_left.y + branch[2].height  {put(&mut branch[2], i, particles);}
            else if branch[3].top_left.x < p.r.x && p.r.x < branch[3].top_left.x + branch[3].width && branch[3].top_left.y < p.r.y && p.r.y < branch[3].top_left.y + branch[3].height  {put(&mut branch[3], i, particles);}
            else {}
        }
    };
}

fn calculate_field(node: &QuadNode, i: usize, particles: &mut Vec<Particle>){
    match &node.qt{
        // empty nodes make no field
        QuadTree::Leaf(None) => {},
        // for a node with single particle, ie leaf, calculate the distance separate
        &QuadTree::Leaf(Some(j)) => {
            if i!=j{
                let r:Vec2 = particles[j].r - particles[i].r;
                let d = (r.length_squared()+SOFTENING*SOFTENING).sqrt();
                let m = particles[j].mass;
                let field = GRAV*m*r/d/d/d;
                particles[i].field += field;
            }
        },
        // if its a branch, either check if the condition for approximation holds true, or recursve
        // through the tree until all particles are included
        QuadTree::Branch(branch) => {
            let m = node.mass;
            let r:Vec2 = node.center_of_mass_sum/m - particles[i].r;
            let d = (r.length_squared() + SOFTENING*SOFTENING).sqrt();
            let s = node.width;
            if s/d <= THETA {
                particles[i].field += GRAV*m*r/d/d/d;
            }
            else {
                calculate_field(&branch[0], i, particles);
                calculate_field(&branch[1], i, particles);
                calculate_field(&branch[2], i, particles);
                calculate_field(&branch[3], i, particles);
            }
        },
    }
}
