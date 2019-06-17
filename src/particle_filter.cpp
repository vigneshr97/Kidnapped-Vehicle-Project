/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;

static default_random_engine rand_gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) 
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if(is_initialized) {
    return;
  } 
  num_particles = 100;  // TODO: Set the number of particles

  normal_distribution<double> x_distribution(x, std[0]);
  normal_distribution<double> y_distribution(y, std[1]);
  normal_distribution<double> theta_distribution(theta, std[2]);

  for (int i = 0; i < num_particles; i++) 
  {
    Particle p;
    p.id = i;
    p.x = x_distribution(rand_gen);
    p.y = y_distribution(rand_gen);
    p.theta = theta_distribution(rand_gen);
    p.weight = 1;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double theta1, theta2, x1, y1;
  for (int i = 0; i < num_particles; ++i)
  {
    if (fabs(yaw_rate) < 0.00001)
    {
      theta1 = particles[i].theta;
      x1 = particles[i].x + (velocity * delta_t * cos(theta1));
      y1 = particles[i].y + (velocity * delta_t * sin(theta1));
    }
    else
    {
      theta1 = particles[i].theta + yaw_rate * delta_t;
      theta2 = particles[i].theta;
      x1 = particles[i].x + (velocity / yaw_rate) * (sin(theta1) - sin(theta2));
      y1 = particles[i].y + (velocity / yaw_rate) * (-1*cos(theta1) + cos(theta2));
    }
    normal_distribution<double> x_distribution(x1, std_pos[0]);
    normal_distribution<double> y_distribution(y1, std_pos[1]);
    normal_distribution<double> theta_distribution(theta1, std_pos[2]);

    particles[i].x =  x_distribution(rand_gen);
    particles[i].y = y_distribution(rand_gen);
    particles[i].theta = theta_distribution(rand_gen);
  }
}


void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, vector<LandmarkObs>& observations) 
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  int obs_size = observations.size();
  int pred_size = predicted.size();
  for (int i = 0; i < obs_size; ++i)
  {
    double minimum_distance = std::numeric_limits<double>::max();
    int best_id = -1;
    for (int j = 0; j < pred_size; ++j)
    {
      if (dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y) < minimum_distance)
      {
        minimum_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
        best_id = predicted[j].id;
      }
    }
    observations[i].id = best_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
  const vector<LandmarkObs> &observations, const Map &map_landmarks) 
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; ++i)
  {
    vector<LandmarkObs> pred;
    int map_size = map_landmarks.landmark_list.size();
    int obs_size = observations.size();

    for (int j = 0; j < map_size; ++j)
    {
      if (fabs(map_landmarks.landmark_list[j].x_f - particles[i].x) <= sensor_range
        && fabs(map_landmarks.landmark_list[j].y_f - particles[i].y) <= sensor_range)
      {
        pred.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }

    vector<LandmarkObs> transformed_obs;
    for (int j = 0; j < obs_size; j++) 
    {
      double tx = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
      double ty = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
      transformed_obs.push_back(LandmarkObs{ observations[j].id, tx, ty });
    }

    dataAssociation(pred, transformed_obs);

    int trans_obs_size = transformed_obs.size();
    int pred_size = pred.size();
    particles[i].weight = 1;
    for (int j = 0; j < trans_obs_size; ++j)
    {
      double pred_x, pred_y;
      for (int k = 0; k < pred_size; ++k)
      {
        if(pred[k].id == transformed_obs[j].id)
        {
          pred_x = pred[k].x;
          pred_y = pred[k].y;
        }
      }
      double coeff = 1/(2*M_PI*std_landmark[0]*std_landmark[1]);
      double term1 = pow(pred_x-transformed_obs[j].x,2)/(2*pow(std_landmark[0], 2));
      double term2 = pow(pred_y-transformed_obs[j].y,2)/(2*pow(std_landmark[1], 2));
      double value = coeff * exp(-term1-term2);
      particles[i].weight *= value;
    }
  }
}

void ParticleFilter::resample() 
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles;
  vector<double> weights;

  std::uniform_int_distribution<int> particle_index(0, num_particles - 1);

  int current_index = particle_index(rand_gen);

  double beta = 0;
  int particle_size = particles.size();

  for (int i = 0; i < num_particles; ++i)
  {
    weights.push_back(particles[i].weight);
  }

  for (int i = 0; i < particle_size; ++i)
  {
    std::uniform_real_distribution<double> random_weight(0, 2* *max_element(weights.begin(), weights.end()));
    beta += random_weight(rand_gen);
    while (beta > weights[current_index]) 
    {
      beta -= weights[current_index];
      current_index = (current_index + 1) % num_particles;
    }
    new_particles.push_back(particles[current_index]);
  }
  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const vector<int>& associations, 
  const vector<double>& sense_x, const vector<double>& sense_y) 
{
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
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
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
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
