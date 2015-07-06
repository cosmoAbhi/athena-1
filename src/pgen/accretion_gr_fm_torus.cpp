// General relativistic Fishbone-Moncrief torus generator

// Primary header
#include "../mesh.hpp"

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // exp(), pow(), sin(), sqrt()
#include <limits>     // numeric_limits::max()

// Athena headers
#include "../athena.hpp"                   // enums, Real
#include "../athena_arrays.hpp"            // AthenaArray
#include "../parameter_input.hpp"          // ParameterInput
#include "../bvals/bvals.hpp"              // BoundaryValues, InterfaceField
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../fluid/fluid.hpp"              // Fluid
#include "../fluid/eos/eos.hpp"            // FluidEqnOfState
#include "../field/field.hpp"              // Field

// TODO: remove with boundary hack
#include <cassert>

// Declarations
void OutflowPrimInnerFluid(MeshBlock *pmb, AthenaArray<Real> &cons,
    int is, int ie, int js, int je, int ks, int ke);
void OutflowPrimOuterFluid(MeshBlock *pmb, AthenaArray<Real> &cons,
    int is, int ie, int js, int je, int ks, int ke);
void FixedInnerField(MeshBlock *pmb, InterfaceField &b,
    int is, int ie, int js, int je, int ks, int ke);
void FixedOuterField(MeshBlock *pmb, InterfaceField &b,
    int is, int ie, int js, int je, int ks, int ke);
static void reset_l_from_r_peak();
static Real log_h_aux(Real r, Real sin_theta);
static void calculate_velocity_in_torus(Real r, Real sin_theta, Real *pu0, Real *pu3);

// Global variables
static Real m, a;                            // black hole parameters
static Real gamma_adi, k_adi;                // fluid parameters
static Real r_edge, r_peak, l, rho_max;      // disk parameters
static Real rho_min, rho_pow, u_min, u_pow;  // background parameters
static Real potential_cutoff;                // sets region of torus to magnetize
static Real beta_min;                        // min ratio of gas to magnetic pressure

// Function for setting initial conditions
// Inputs:
//   pfl: Fluid
//   pfd: Field (unused)
//   pin: parameters
// Outputs: (none)
// Notes:
//   initializes Fishbone-Moncrief torus
//     sets both primitive and conserved variables
//   defines and enrolls fixed r- and theta-direction boundary conditions
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//              Fishbone 1977, ApJ 215 323 (F)
//   assumes x3 is axisymmetric direction
void Mesh::ProblemGenerator(Fluid *pfl, Field *pfd, ParameterInput *pin)
{
  // Prepare index bounds
  MeshBlock *pb = pfl->pmy_block;
  int il = pb->is - NGHOST;
  int iu = pb->ie + NGHOST;
  int jl = pb->js;
  int ju = pb->je;
  if (pb->block_size.nx2 > 1)
  {
    jl -= (NGHOST);
    ju += (NGHOST);
  }
  int kl = pb->ks;
  int ku = pb->ke;
  if (pb->block_size.nx3 > 1)
  {
    kl -= (NGHOST);
    ku += (NGHOST);
  }

  // Get mass and spin of black hole
  m = pb->pcoord->GetMass();
  a = pb->pcoord->GetSpin();

  // Get ratio of specific heats
  gamma_adi = pfl->pf_eos->GetGamma();

  // Read other properties
  rho_min = pin->GetReal("fluid", "rho_min");
  rho_pow = pin->GetReal("fluid", "rho_pow");
  u_min = pin->GetReal("fluid", "u_min");
  u_pow = pin->GetReal("fluid", "u_pow");
  k_adi = pin->GetReal("problem", "k_adi");
  r_edge = pin->GetReal("problem", "r_edge");
  r_peak = pin->GetReal("problem", "r_peak");
  l = pin->GetReal("problem", "l");
  rho_max = pin->GetReal("problem", "rho_max");
  potential_cutoff = pin->GetReal("problem", "cutoff");
  beta_min = pin->GetReal("problem", "beta_min");

  // Reset l if valid r_peak given
  if (r_peak >= 0.0)
    reset_l_from_r_peak();

  // Prepare scratch arrays
  AthenaArray<bool> in_torus;
  in_torus.NewAthenaArray(ju+1, iu+1);
  AthenaArray<Real> g, g_inv;
  g.NewAthenaArray(NMETRIC, iu+1);
  g_inv.NewAthenaArray(NMETRIC, iu+1);

  // Initialize primitive values
  Real log_h_edge = log_h_aux(r_edge, 1.0);
  Real rho_peak = 0.0;
  for (int j = jl; j <= ju; j++)
  {
    pb->pcoord->CellMetric(kl, j, il, iu, g, g_inv);
    for (int i = il; i <= iu; i++)
    {
      // Get Boyer-Lindquist coordinates of cell
      Real r, theta, phi;
      pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i), pb->pcoord->x2v(j),
          pb->pcoord->x3v(kl), &r, &theta, &phi);
      Real sin_theta = std::sin(theta);
      Real alpha_sq = -1.0 / g_inv(I00,i);

      // Determine if we are in the torus
      Real log_h;
      in_torus(j,i) = false;
      if (r >= r_edge)
      {
        log_h = log_h_aux(r, sin_theta) - log_h_edge;  // (FM 3.6)
        if (log_h >= 0.0)
          in_torus(j,i) = true;
      }

      // Calculate primitives depending on location
      Real rho, pgas, unorm1, unorm2, unorm3;
      if (in_torus(j,i))
      {
        Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1);
        rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0));
        pgas = pgas_over_rho * rho;
        rho_peak = std::max(rho_peak, rho);
        Real u0, u3;
        calculate_velocity_in_torus(r, sin_theta, &u0, &u3);
        Real u0_pref, u1_pref, u2_pref, u3_pref;
        pb->pcoord->TransformVectorCell(u0, 0.0, 0.0, u3, kl, j, i,
            &u0_pref, &u1_pref, &u2_pref, &u3_pref);
        unorm1 = u1_pref + alpha_sq * g_inv(I01,i) * u0_pref;
        unorm2 = u2_pref + alpha_sq * g_inv(I02,i) * u0_pref;
        unorm3 = u3_pref + alpha_sq * g_inv(I03,i) * u0_pref;
      }
      else
      {
        rho = rho_min * std::pow(r, rho_pow);
        Real u = u_min * std::pow(r, u_pow);
        pgas = (gamma_adi-1.0) * u;
        unorm1 = 0.0;
        unorm2 = 0.0;
        unorm3 = 0.0;
      }

      // Set primitive values
      for (int k = kl; k <= ku; k++)
      {
        pfl->w(IDN,k,j,i) = pfl->w1(IDN,k,j,i) = rho;
        pfl->w(IEN,k,j,i) = pfl->w1(IEN,k,j,i) = pgas;
        pfl->w(IVX,k,j,i) = pfl->w1(IM1,k,j,i) = unorm1;
        pfl->w(IVY,k,j,i) = pfl->w1(IM2,k,j,i) = unorm2;
        pfl->w(IVZ,k,j,i) = pfl->w1(IM3,k,j,i) = unorm3;
      }
    }
  }

  // Initialize magnetic fields
  if (MAGNETIC_FIELDS_ENABLED)
  {
    // Prepare 2D arrays of vector potential values
    AthenaArray<Real> a_phi_cells, a_phi_edges;
    a_phi_cells.NewAthenaArray(ju+1, iu+1);
    a_phi_edges.NewAthenaArray(ju+2, iu+2);

    // Go through 2D slice, setting vector potential in cells
    for (int j = jl; j <= ju; j++)
      for (int i = il; i <= iu; i++)
      {
        // Get Boyer-Lindquist coordinates
        Real r, theta, phi;
        pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i), pb->pcoord->x2v(j),
            pb->pcoord->x3v(kl), &r, &theta, &phi);
        Real sin_theta = std::sin(theta);

        // Calculate A_phi as proportional to rho
        if (r >= r_edge)
        {
          Real log_h = log_h_aux(r, sin_theta) - log_h_edge;  // (FM 3.6)
          if (log_h >= 0.0)
          {
            Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1);
            Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0));
            rho_peak = std::max(rho_peak, rho);
            a_phi_cells(j,i) = rho;
          }
        }
      }

    // Go through 2D slice, setting vector potential at edges
    for (int j = jl; j <= ju+1; j++)
      for (int i = il; i <= iu+1; i++)
      {
        // Get Boyer-Lindquist coordinates
        Real r, theta, phi;
        pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1f(i), pb->pcoord->x2f(j),
            pb->pcoord->x3v(kl), &r, &theta, &phi);

        // Calculate A_phi as proportional to rho
        if (r >= r_edge)
        {
          Real sin_theta = std::sin(theta);
          Real log_h = log_h_aux(r, sin_theta) - log_h_edge;  // (FM 3.6)
          if (log_h >= 0.0)
          {
            Real pgas_over_rho = (gamma_adi-1.0)/gamma_adi * (std::exp(log_h)-1);
            Real rho = std::pow(pgas_over_rho/k_adi, 1.0/(gamma_adi-1.0));
            rho_peak = std::max(rho_peak, rho);
            a_phi_edges(j,i) = rho;
          }
        }
      }

    // Normalize vector potential
    for (int j = jl; j <= ju; j++)
      for (int i = il; i <= iu; i++)
        a_phi_cells(j,i) = std::max(a_phi_cells(j,i)/rho_peak - potential_cutoff, 0.0);
    for (int j = jl; j <= ju+1; j++)
      for (int i = il; i <= iu+1; i++)
        a_phi_edges(j,i) = std::max(a_phi_edges(j,i)/rho_peak - potential_cutoff, 0.0);

    // Set magnetic fields according to vector potential
    // Note: This does very rough differencing for in-face fields on exterior faces of
    //    domain. This should not matter, as these will be identically 0 in nice
    //    coordinate systems or as long as the initial torus is within the domain.
    for (int k = kl; k <= ku+1; ++k)
      for (int j = jl; j <= ju+1; ++j)
        for (int i = il; i <= iu+1; ++i)
        {
          // Set B^1
          if (j != ju+1 and k != ku+1)
          {
            Real r, theta, phi;
            pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1f(i),
                pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r, &theta, &phi);
            Real r_1, theta_1, phi_1;
            pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1f(i),
                pb->pcoord->x2f(j), pb->pcoord->x3v(k), &r_1, &theta_1, &phi_1);
            Real r_2, theta_2, phi_2;
            pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1f(i),
                pb->pcoord->x2f(j+1), pb->pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            Real bbr = (a_phi_edges(j+1,i) - a_phi_edges(j,i)) / (theta_2 - theta_1);
            Real a_phi_1, a_phi_2;
            if (i == il)
            {
              a_phi_1 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j+1,i));
              a_phi_2 = a_phi_cells(j,i);
              r_1 = r;
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                  pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            }
            else if (i == iu+1)
            {
              a_phi_1 = a_phi_cells(j,i-1);
              a_phi_2 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j+1,i));
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i-1),
                  pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              r_2 = r;
            }
            else
            {
              a_phi_1 = a_phi_cells(j,i-1);
              a_phi_2 = a_phi_cells(j,i);
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i-1),
                  pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                  pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            }
            Real bbtheta = (a_phi_2 - a_phi_1) / (r_2 - r_1);
            if (bbr == 0.0 and bbtheta == 0.0)
              pfd->b.x1f(k,j,i) = 0.0;
            else
            {
              Real ut, uphi;
              Real sin_theta = std::sin(theta);
              calculate_velocity_in_torus(r, sin_theta, &ut, &uphi);
              Real sin_sq_theta = SQR(sin_theta);
              Real cos_sq_theta = 1.0 - sin_sq_theta;
              Real rho_sq = SQR(r) + SQR(a) * cos_sq_theta;
              Real bt = -2.0*m*a*r * SQR(sin_theta) / rho_sq * bbr * ut;
              Real br = 1.0/ut * bbr;
              Real btheta = 1.0/ut * bbtheta;
              Real u0, u1, u2, u3;
              pb->pcoord->TransformVectorFace1(ut, 0.0, 0.0, uphi, k, j, i,
                  &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              pb->pcoord->TransformVectorFace1(bt, br, btheta, 0.0, k, j, i,
                  &b0, &b1, &b2, &b3);
              pfd->b.x1f(k,j,i) = b1 * u0 - b0 * u1;
            }
          }

          // Set B^2
          if (i != iu+1 and k != ku+1)
          {
            Real r, theta, phi;
            pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                pb->pcoord->x2f(j), pb->pcoord->x3v(k), &r, &theta, &phi);
            Real r_1, theta_1, phi_1;
            pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1f(i),
                pb->pcoord->x2f(j), pb->pcoord->x3v(k), &r_1, &theta_1, &phi_1);
            Real r_2, theta_2, phi_2;
            pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1f(i+1),
                pb->pcoord->x2f(j), pb->pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            Real bbtheta = (a_phi_edges(j,i+1) - a_phi_edges(j,i)) / (r_2 - r_1);
            Real a_phi_1, a_phi_2;
            if (j == jl)
            {
              a_phi_1 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j,i+1));
              a_phi_2 = a_phi_cells(j,i);
              theta_1 = theta;
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                  pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            }
            else if (j == ju+1)
            {
              a_phi_1 = a_phi_cells(j-1,i);
              a_phi_2 = 0.5 * (a_phi_edges(j,i) + a_phi_edges(j,i+1));
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                  pb->pcoord->x2v(j-1), pb->pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              theta_2 = theta;
            }
            else
            {
              a_phi_1 = a_phi_cells(j-1,i);
              a_phi_2 = a_phi_cells(j,i);
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                  pb->pcoord->x2v(j-1), pb->pcoord->x3v(k), &r_1, &theta_1, &phi_1);
              pb->pcoord->GetBoyerLindquistCoordinates(pb->pcoord->x1v(i),
                  pb->pcoord->x2v(j), pb->pcoord->x3v(k), &r_2, &theta_2, &phi_2);
            }
            Real bbr = (a_phi_2 - a_phi_1) / (theta_2 - theta_1);
            if (bbr == 0.0 and bbtheta == 0.0)
              pfd->b.x2f(k,j,i) = 0.0;
            else
            {
              Real ut, uphi;
              Real sin_theta = std::sin(theta);
              calculate_velocity_in_torus(r, sin_theta, &ut, &uphi);
              Real sin_sq_theta = SQR(sin_theta);
              Real cos_sq_theta = 1.0 - sin_sq_theta;
              Real rho_sq = SQR(r) + SQR(a) * cos_sq_theta;
              Real bt = -2.0*m*a*r * SQR(sin_theta) / rho_sq * bbr * ut;
              Real br = 1.0/ut * bbr;
              Real btheta = 1.0/ut * bbtheta;
              Real u0, u1, u2, u3;
              pb->pcoord->TransformVectorFace2(ut, 0.0, 0.0, uphi, k, j, i,
                  &u0, &u1, &u2, &u3);
              Real b0, b1, b2, b3;
              pb->pcoord->TransformVectorFace2(bt, br, btheta, 0.0, k, j, i,
                  &b0, &b1, &b2, &b3);
              pfd->b.x2f(k,j,i) = b2 * u0 - b0 * u2;
            }
          }

          // Set B^3
          if (i != iu+1 and j != ju+1)
            pfd->b.x3f(k,j,i) = 0.0;
        }
    a_phi_cells.DeleteAthenaArray();
    a_phi_edges.DeleteAthenaArray();
  }

  // Normalize density and pressure
  if (rho_max > 0.0)
    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
        for (int i = il; i <= iu; ++i)
          if (in_torus(j,i))
          {
            pfl->w(IDN,k,j,i) /= rho_peak;
            pfl->w1(IDN,k,j,i) /= rho_peak;
            pfl->w(IEN,k,j,i) /= rho_peak;
            pfl->w1(IEN,k,j,i) /= rho_peak;
          }

  // Impose density and pressure floors
  for (int k = kl; k <= ku; ++k)
    for (int j = jl; j <= ju; ++j)
      for (int i = il; i <= iu; ++i)
      {
        Real r, theta, phi;
        pb->pcoord->GetBoyerLindquistCoordinates(pb->x1v(i), pb->x2v(j), pb->x3v(kl),
            &r, &theta, &phi);
        Real &rho = pfl->w(IDN,k,j,i);
        Real &pgas = pfl->w(IEN,k,j,i);
        rho = std::max(rho, rho_min * std::pow(r, rho_pow));
        pgas = std::max(pgas, (gamma_adi-1.0) * u_min * std::pow(r, u_pow));
        pfl->w1(IDN,k,j,i) = rho;
        pfl->w1(IEN,k,j,i) = pgas;
      }

  // Calculate cell-centered magnetic field
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ku+1, ju+1, iu+1);
  if (MAGNETIC_FIELDS_ENABLED)
    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
        for (int i = il; i <= iu; ++i)
        {
          // Extract face-centered magnetic field
          const Real &bbf1m = pfd->b.x1f(k,j,i);
          const Real &bbf1p = pfd->b.x1f(k,j,i+1);
          const Real &bbf2m = pfd->b.x2f(k,j,i);
          const Real &bbf2p = pfd->b.x2f(k,j+1,i);
          const Real &bbf3m = pfd->b.x3f(k,j,i);
          const Real &bbf3p = pfd->b.x3f(k+1,j,i);

          // Calculate cell-centered magnetic field
          Real tmp = (pb->pcoord->x1v(i) - pb->pcoord->x1f(i)) / pb->pcoord->dx1f(i);
          bb(IB1,k,j,i) = (1.0-tmp) * bbf1m + tmp * bbf1p;
          tmp = (pb->pcoord->x2v(j) - pb->pcoord->x2f(j)) / pb->pcoord->dx2f(j);
          bb(IB2,k,j,i) = (1.0-tmp) * bbf2m + tmp * bbf2p;
          tmp = (pb->pcoord->x3v(k) - pb->pcoord->x3f(k)) / pb->pcoord->dx3f(k);
          bb(IB3,k,j,i) = (1.0-tmp) * bbf3m + tmp * bbf3p;
       }

  // Calculate minimum beta
  Real beta_actual = std::numeric_limits<Real>::max();
  if (MAGNETIC_FIELDS_ENABLED)
    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
      {
        pb->pcoord->CellMetric(k, j, il, iu, g, g_inv);
        for (int i = il; i <= iu; ++i)
        {
          Real alpha = std::sqrt(-1.0/g_inv(I00,i));
          Real unorm1 = pfl->w(IVX,k,j,i);
          Real unorm2 = pfl->w(IVY,k,j,i);
          Real unorm3 = pfl->w(IVZ,k,j,i);
          Real tmp = g(I11,i)*unorm1*unorm1 + 2.0*g(I12,i)*unorm1*unorm2
                       + 2.0*g(I13,i)*unorm1*unorm3
                   + g(I22,i)*unorm2*unorm2 + 2.0*g(I23,i)*unorm2*unorm3
                   + g(I33,i)*unorm3*unorm3;
          Real gamma = std::sqrt(1.0 + tmp);
          Real u0 = gamma / alpha;
          Real u1 = unorm1 - alpha * gamma * g_inv(I01,i);
          Real u2 = unorm2 - alpha * gamma * g_inv(I02,i);
          Real u3 = unorm3 - alpha * gamma * g_inv(I03,i);
          Real bb1 = bb(IB1,k,j,i);
          Real bb2 = bb(IB2,k,j,i);
          Real bb3 = bb(IB3,k,j,i);
          Real b0 =
                g(I01,i)*bb1*u0 + g(I11,i)*bb1*u1 + g(I12,i)*bb1*u2 + g(I13,i)*bb1*u3
              + g(I02,i)*bb2*u0 + g(I12,i)*bb2*u1 + g(I22,i)*bb2*u2 + g(I23,i)*bb2*u3
              + g(I03,i)*bb3*u0 + g(I13,i)*bb3*u1 + g(I23,i)*bb3*u2 + g(I33,i)*bb3*u3;
          Real b1 = 1.0/u0 * (bb1 + b0 * u1);
          Real b2 = 1.0/u0 * (bb2 + b0 * u2);
          Real b3 = 1.0/u0 * (bb3 + b0 * u3);
          Real b_0, b_1, b_2, b_3;
          pb->pcoord->LowerVectorCell(b0, b1, b2, b3, k, j, i, &b_0, &b_1, &b_2, &b_3);
          Real b_sq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;
          if (b_sq > 0.0)
            beta_actual = std::min(beta_actual, pfl->w(IEN,k,j,i)/(b_sq/2.0));
        }
      }

  // Free scratch arrays
  in_torus.DeleteAthenaArray();
  g.DeleteAthenaArray();
  g_inv.DeleteAthenaArray();

  // Normalize magnetic field to have desired beta
  if (MAGNETIC_FIELDS_ENABLED)
  {
    Real normalization = std::sqrt(beta_actual/beta_min);
    for (int k = kl; k <= ku+1; ++k)
      for (int j = jl; j <= ju+1; ++j)
        for (int i = il; i <= iu+1; ++i)
        {
          if (j != ju+1 and k != ku+1)
            pfd->b.x1f(k,j,i) *= normalization;
          if (i != iu+1 and k != ku+1)
            pfd->b.x2f(k,j,i) *= normalization;
          if (i != iu+1 and j != ju+1)
            pfd->b.x3f(k,j,i) *= normalization;
        }
    for (int k = kl; k <= ku; ++k)
      for (int j = jl; j <= ju; ++j)
        for (int i = il; i <= iu; ++i)
        {
          bb(IB1,k,j,i) *= normalization;
          bb(IB2,k,j,i) *= normalization;
          bb(IB3,k,j,i) *= normalization;
        }
  }

  // Initialize conserved values
  pb->pfluid->pf_eos->PrimitiveToConserved(pfl->w, bb, pfl->u);  
  bb.DeleteAthenaArray();

  // Enroll boundary functions
  pb->pbval->EnrollFluidBoundaryFunction(inner_x1, OutflowPrimInnerFluid);
  pb->pbval->EnrollFluidBoundaryFunction(outer_x1, OutflowPrimOuterFluid);
  if (MAGNETIC_FIELDS_ENABLED)
  {
    pb->pbval->EnrollFieldBoundaryFunction(inner_x1, FixedInnerField);
    pb->pbval->EnrollFieldBoundaryFunction(outer_x1, FixedOuterField);
  }
  return;
}

// Inner fluid boundary condition
// Inputs:
//   pmb: pointer to block
// Outputs:
//   cons: conserved quantities set along inner x1-boundary
// Notes:
//   TODO: remove prim hack
//   TODO: only works in hydro
void OutflowPrimInnerFluid(MeshBlock *pmb, AthenaArray<Real> &cons,
    int is, int ie, int js, int je, int ks, int ke)
{
  AthenaArray<Real> *pprim;
  if (&cons == &pmb->pfluid->u)
    pprim = &pmb->pfluid->w;
  else if (&cons == &pmb->pfluid->u1)
    pprim = &pmb->pfluid->w1;
  else
    assert(0);
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC,is);
  gi.NewAthenaArray(NMETRIC,is);
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
    {
      pmb->pcoord->CellMetric(k, j, is-NGHOST, is-1, g, gi);
      Real unorm1 = (*pprim)(IVX,k,j,is);
      Real unorm2 = (*pprim)(IVY,k,j,is);
      Real unorm3 = (*pprim)(IVZ,k,j,is);
      for (int i = is-NGHOST; i <= is-1; ++i)
      {
        Real &rho = (*pprim)(IDN,k,j,i);
        Real &pgas = (*pprim)(IEN,k,j,i);
        rho = (*pprim)(IDN,k,j,is);
        pgas = (*pprim)(IEN,k,j,is);
        (*pprim)(IVX,k,j,i) = unorm1;
        (*pprim)(IVY,k,j,i) = unorm2;
        (*pprim)(IVZ,k,j,i) = unorm3;
        Real alpha = std::sqrt(-1.0/gi(I00,i));
        Real tmp = g(I11,i)*unorm1*unorm1 + 2.0*g(I12,i)*unorm1*unorm2
                     + 2.0*g(I13,i)*unorm1*unorm3
                 + g(I22,i)*unorm2*unorm2 + 2.0*g(I23,i)*unorm2*unorm3
                 + g(I33,i)*unorm3*unorm3;
        Real gamma = std::sqrt(1.0 + tmp);
        Real u0 = gamma / alpha;
        Real u1 = unorm1 - alpha * gamma * gi(I01,i);
        Real u2 = unorm2 - alpha * gamma * gi(I02,i);
        Real u3 = unorm3 - alpha * gamma * gi(I03,i);
        Real u_0, u_1, u_2, u_3;
        pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);
        Real gamma_adi = pmb->pfluid->pf_eos->GetGamma();
        Real gamma_prime = gamma_adi/(gamma_adi-1.0);
        Real wgas = rho + gamma_prime * pgas;
        cons(IDN,k,j,i) = rho * u0;
        cons(IEN,k,j,i) = wgas * u0 * u_0 + pgas;
        cons(IM1,k,j,i) = wgas * u0 * u_1;
        cons(IM2,k,j,i) = wgas * u0 * u_2;
        cons(IM3,k,j,i) = wgas * u0 * u_3;
      }
    }
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();
  return;
}

// Outer fluid boundary condition
// Inputs:
//   pmb: pointer to block
// Outputs:
//   cons: conserved quantities set along outer x1-boundary
// Notes:
//   TODO: remove prim hack
//   TODO: only works in hydro
void OutflowPrimOuterFluid(MeshBlock *pmb, AthenaArray<Real> &cons,
    int is, int ie, int js, int je, int ks, int ke)
{
  AthenaArray<Real> *pprim;
  if (&cons == &pmb->pfluid->u)
    pprim = &pmb->pfluid->w;
  else if (&cons == &pmb->pfluid->u1)
    pprim = &pmb->pfluid->w1;
  else
    assert(0);
  AthenaArray<Real> g, gi;
  g.NewAthenaArray(NMETRIC,ie+NGHOST+1);
  gi.NewAthenaArray(NMETRIC,ie+NGHOST+1);
  for (int k = ks; k <= ke; ++k)
    for (int j = js; j <= je; ++j)
    {
      pmb->pcoord->CellMetric(k, j, ie+1, ie+NGHOST, g, gi);
      Real unorm1 = (*pprim)(IVX,k,j,ie);
      Real unorm2 = (*pprim)(IVY,k,j,ie);
      Real unorm3 = (*pprim)(IVZ,k,j,ie);
      for (int i = ie+1; i <= ie+NGHOST; ++i)
      {
        Real &rho = (*pprim)(IDN,k,j,i);
        Real &pgas = (*pprim)(IEN,k,j,i);
        rho = (*pprim)(IDN,k,j,ie);
        pgas = (*pprim)(IEN,k,j,ie);
        (*pprim)(IVX,k,j,i) = unorm1;
        (*pprim)(IVY,k,j,i) = unorm2;
        (*pprim)(IVZ,k,j,i) = unorm3;
        Real alpha = std::sqrt(-1.0/gi(I00,i));
        Real tmp = g(I11,i)*unorm1*unorm1 + 2.0*g(I12,i)*unorm1*unorm2
                     + 2.0*g(I13,i)*unorm1*unorm3
                 + g(I22,i)*unorm2*unorm2 + 2.0*g(I23,i)*unorm2*unorm3
                 + g(I33,i)*unorm3*unorm3;
        Real gamma = std::sqrt(1.0 + tmp);
        Real u0 = gamma / alpha;
        Real u1 = unorm1 - alpha * gamma * gi(I01,i);
        Real u2 = unorm2 - alpha * gamma * gi(I02,i);
        Real u3 = unorm3 - alpha * gamma * gi(I03,i);
        Real u_0, u_1, u_2, u_3;
        pmb->pcoord->LowerVectorCell(u0, u1, u2, u3, k, j, i, &u_0, &u_1, &u_2, &u_3);
        Real gamma_adi = pmb->pfluid->pf_eos->GetGamma();
        Real gamma_prime = gamma_adi/(gamma_adi-1.0);
        Real wgas = rho + gamma_prime * pgas;
        cons(IDN,k,j,i) = rho * u0;
        cons(IEN,k,j,i) = wgas * u0 * u_0 + pgas;
        cons(IM1,k,j,i) = wgas * u0 * u_1;
        cons(IM2,k,j,i) = wgas * u0 * u_2;
        cons(IM3,k,j,i) = wgas * u0 * u_3;
      }
    }
  g.DeleteAthenaArray();
  gi.DeleteAthenaArray();
  return;
}

// Inner field boundary condition
// Inputs:
//   pmb: pointer to block
// Outputs:
//   b: magnetic field
// Notes:
//   does nothing
void FixedInnerField(MeshBlock *pmb, InterfaceField &b,
    int is, int ie, int js, int je, int ks, int ke)
{
  return;
}

// Outer field boundary condition
// Inputs:
//   pmb: pointer to block
// Outputs:
//   b: magnetic field
// Notes:
//   does nothing
void FixedOuterField(MeshBlock *pmb, InterfaceField &b,
    int is, int ie, int js, int je, int ks, int ke)
{
  return;
}

// Function for calculating angular momentum variable l
// Inputs: (none)
// Outputs:
//   sets global variable l = u^t u_\phi such that pressure maximum occurs at r_peak
// Notes:
//   beware many different definitions of l abound
//     this is *not* -u_phi/u_t
//   Harm has a similar function: lfish_calc() in init.c
//     Harm's function assumes M = 1 and that corotation is desired
//     it is equivalent to this, though seeing this requires much manipulation
//   implements (3.8) from Fishbone & Moncrief 1976, ApJ 207 962
//   assumes corotation
//   TODO: add counterrotation option
static void reset_l_from_r_peak()
{
  Real num = SQR(SQR(r_peak)) + SQR(a*r_peak) - 2.0*m*SQR(a)*r_peak
      - a*(SQR(r_peak)-SQR(a))*std::sqrt(m*r_peak);
  Real denom = SQR(r_peak) - 3.0*m*r_peak + 2.0*a*std::sqrt(m*r_peak);
  l = 1.0/r_peak * std::sqrt(m/r_peak) * num/denom;
  return;
}

// Function for helping to calculate enthalpy
// Inputs:
//   r: radial Boyer-Lindquist coordinate
//   sin_sq_theta: square of sine of polar Boyer-Lindquist coordinate
// Outputs:
//   returned value: log(h)
// Notes:
//   enthalpy defined here as h = p_gas/rho
//   references Fishbone & Moncrief 1976, ApJ 207 962 (FM)
//   implements first half of (FM 3.6)
static Real log_h_aux(Real r, Real sin_theta)
{
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*m*r + SQR(a);                // \Delta
  Real sigma = SQR(r) + SQR(a)*cos_sq_theta;             // \Sigma
  Real aa = SQR(SQR(r)+SQR(a))
      - delta*SQR(a)*sin_sq_theta;                       // A
  Real exp_2nu = sigma * delta / aa;                     // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;             // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;                 // \exp(-2\chi) (cf. FM 2.15)
  Real omega = 2.0*m*a*r/aa;                             // \omega (FM 3.5)
  Real var_a = std::sqrt(1.0 + 4.0*SQR(l)*exp_neg2chi);
  Real var_b = 0.5 * std::log((1.0+var_a)
      / (sigma*delta/aa));
  Real var_c = -0.5 * var_a;
  Real var_d = -l * omega;
  return var_b + var_c + var_d;                          // (FM 3.4)
}

// Function for computing 4-velocity components at a given position inside torus
// Inputs:
//   r: Boyer-Lindquist r
//   sin_theta: sine of Boyer-Lindquist theta
// Outputs:
//   pu0: u^t set (Boyer-Lindquist coordinates)
//   pu3: u^\phi set (Boyer-Lindquist coordinates)
// Notes:
//   The formula for u^3 as a function of u_{(\phi)} is tedious to derive,
//       but this matches the formula used in Harm (init.c).
static void calculate_velocity_in_torus(Real r, Real sin_theta, Real *pu0, Real *pu3)
{
  Real sin_sq_theta = SQR(sin_theta);
  Real cos_sq_theta = 1.0 - sin_sq_theta;
  Real delta = SQR(r) - 2.0*m*r + SQR(a);            // \Delta
  Real sigma = SQR(r) + SQR(a)*cos_sq_theta;         // \Sigma
  Real aa = SQR(SQR(r)+SQR(a))
      - delta*SQR(a)*sin_sq_theta;                   // A
  Real exp_2nu = sigma * delta / aa;                 // \exp(2\nu) (FM 3.5)
  Real exp_2psi = aa / sigma * sin_sq_theta;         // \exp(2\psi) (FM 3.5)
  Real exp_neg2chi = exp_2nu / exp_2psi;             // \exp(-2\chi) (cf. FM 2.15)
  Real u_phi_proj_a = 1.0 + 4.0*SQR(l)*exp_neg2chi;
  Real u_phi_proj_b = -1.0
      + std::sqrt(u_phi_proj_a);
  Real u_phi_proj = std::sqrt(0.5 * u_phi_proj_b);   // (FM 3.3)
  Real u_3 = std::sqrt(aa/sigma) * sin_theta
      * u_phi_proj;                                  // (FM 2.12, F 2.5, FM 3.5)
  Real u3_a = (1.0+SQR(u_phi_proj))
      / (aa*sigma*delta);
  Real u3_b = 2.0*m*a*r * std::sqrt(u3_a);
  Real u3_c = std::sqrt(sigma/aa) / sin_theta;
  Real u3 = u3_b + u3_c * u_phi_proj;
  Real g_00 = -(1.0 - 2.0*m*r/sigma);
  Real g_03 = -2.0*m*a*r/sigma * sin_sq_theta;
  Real g_33 = (sigma + (1.0 + 2.0*m*r/sigma)
      * SQR(a) * sin_sq_theta) * sin_sq_theta;
  Real u0_a = (SQR(g_03) - g_00*g_33) * SQR(u3);
  Real u0_b = std::sqrt(u0_a - g_00);
  Real u0 = -1.0/g_00 * (g_03*u3 + u0_b);
  *pu0 = u0;
  *pu3 = u3;
  return;
}
