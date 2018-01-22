/*
 * particle_filter.cpp
 *
 *  Created on: Jan 21, 2018
 *      Author: Bob Lutz
 */

#include <random>
#include <iostream>
#include <sstream>

#include "particle_filter.h"

using namespace std;

const int N_PARTICLES = 20;
const double DEFAULT_WEIGHT = 1.0;
const double EPSILON = 0.001;

default_random_engine gen;

struct Point { double x, y; };

bool compare_doubles(double d1, double d2) {
  return (fabs(d2-d1) < EPSILON ? true : false);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles = N_PARTICLES;

  normal_distribution<double> dist_x(x, std[0]), // This block creates a Gaussian
                              dist_y(y, std[1]), // distribution for x,y and theta
                              dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i)
    particles.push_back((Particle){i, dist_x(gen), dist_y(gen), dist_theta(gen), DEFAULT_WEIGHT});
  weights.resize(num_particles, DEFAULT_WEIGHT);
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  normal_distribution<double> n_x(0.0, std_pos[0]),
                              n_y(0.0, std_pos[1]),
                              n_theta(0.0, std_pos[2]);

  double k = velocity * (compare_doubles(yaw_rate, 0.0) ? delta_t : (1.0 / yaw_rate));

  for (int i = 0; i < particles.size(); i++) {

    double noise_x = n_x(gen), noise_y = n_y(gen), noise_theta = n_theta(gen);

    if (compare_doubles(yaw_rate, 0.0)) {
      particles[i].x += k * cos(particles[i].theta) + noise_x;
      particles[i].y += k * sin(particles[i].theta) + noise_y;
      particles[i].theta += noise_theta;
    } else {
      double phi = particles[i].theta + yaw_rate * delta_t;
      particles[i].x += k * (sin(phi) - sin(particles[i].theta)) + noise_x;
      particles[i].y += k * (cos(particles[i].theta) - cos(phi)) + noise_y;
      particles[i].theta = phi + noise_theta;
    }

  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++) {

    int select = 0;
    double min_error = numeric_limits<double>::max();

    for (int j = 0; j < predicted.size(); j++) {
      Point d = (Point){predicted[j].x - observations[i].x, predicted[j].y - observations[i].y};
      double error = d.x * d.x + d.y * d.y;
      if (error < min_error) {
        select = j;
        min_error = error;
      }
    }
    observations[i].id = select;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  Point std = (Point){std_landmark[0], std_landmark[1]};  // used later for calculating the new weights
  double n_a = 0.5 / (std.x * std.x);
  double n_b = 0.5 / (std.y * std.y);

  for (int  i = 0; i < N_PARTICLES; i++) {

    Point p = (Point){particles[i].x,particles[i].y};
    double ptheta = particles[i].theta;
    vector<LandmarkObs> landmarks_in_range, map_observations;

    // transform observations (particle space) to map coordinate space
    for (int j = 0; j < observations.size(); j++){
      Point o = (Point){observations[j].x, observations[j].y};
      Point o_map = (Point){p.x + o.x * cos(ptheta) - o.y * sin(ptheta),
                            p.y + o.y * cos(ptheta) + o.x * sin(ptheta)};
      map_observations.push_back((LandmarkObs){observations[j].id, o_map.x, o_map.y});
    }

    // select landmarks within range
    for (int j = 0;  j < map_landmarks.landmark_list.size(); j++) {
      Point m = (Point){map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f};
      Point d = (Point){m.x - p.x, m.y - p.y};
      double error = sqrt(d.x * d.x + d.y * d.y);
      if (error < sensor_range)
        landmarks_in_range.push_back((LandmarkObs){map_landmarks.landmark_list[j].id_i, m.x, m.y});
    }

    // for landmarks in range, associate these to landmark observations
    dataAssociation(landmarks_in_range, map_observations);

    // compare each vehicle observation to each particle observation in range and update weights
    double w = DEFAULT_WEIGHT;

    for (int j = 0; j < map_observations.size(); j++){ // map_observations in map coordinates

      Point o = (Point){map_observations[j].x, map_observations[j].y};
      Point predicted = (Point){landmarks_in_range[map_observations[j].id].x, // in map coords too
                                landmarks_in_range[map_observations[j].id].y
      };
      Point d = (Point){o.x - predicted.x, o.y - predicted.y};

      double a = n_a * d.x * d.x;
      double b = n_b * d.y * d.y;
      double r = exp(-(a + b)) / sqrt( 2.0 * M_PI * std.x * std.y);
      w *= r;
    }

    particles[i].weight = w;
    weights[i] = w;
  }

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<Particle> v;
  discrete_distribution<> selection(weights.begin(), weights.end());
  for (int n = 0; n < particles.size(); n++) {
    int i = selection(gen);
    v.push_back((Particle){i, particles[i].x, particles[i].y, particles[i].theta, DEFAULT_WEIGHT});
  }
  particles = v;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
