#pragma once
#include <complex>
#include <memory>
#include <vector>

#include <Eigen/Dense>

namespace qunn
{
namespace detail
{
constexpr uint32_t set(uint32_t bitstring, uint32_t idx)
{
	return bitstring | (1u << idx);
}
constexpr uint32_t unset(uint32_t bitstring, uint32_t idx)
{
	return bitstring & (~(1u << idx));
}
}//namespace detail

/* apply single qubit gate m to pos */
template<typename T>
Eigen::VectorXcd apply_single_qubit(const Eigen::VectorXcd& vec,
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, uint32_t pos)
{
	Eigen::VectorXcd res(vec.size());
	for(uint32_t k = 0; k < vec.size(); ++k)
	{
		uint32_t k0 = (k >> pos) & 1u;
		uint32_t i0 = k & (~(1u << pos));
		uint32_t i1 = k | (1u << pos);
		res(k) = m(k0, 0) * vec(i0) + m(k0, 1) * vec(i1);
	}
	return res;
}

template<typename T>
Eigen::VectorXcd apply_two_qubit(const Eigen::VectorXcd& vec,
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, std::array<uint32_t,2> pos)
{
	using detail::set;
	using detail::unset;
	Eigen::VectorXcd res(vec.size());
	for(uint32_t n = 0; n < vec.size(); ++n)
	{
		uint32_t k = (((n >> pos[1]) & 1u) << 1) | ((n >> pos[0]) & 1u);
		uint32_t i0 = unset(unset(n,pos[0]),pos[1]); //0b00
		uint32_t i1 = unset(  set(n,pos[0]),pos[1]); //0b01
		uint32_t i2 =   set(unset(n,pos[0]),pos[1]); //0b10
		uint32_t i3 =   set(  set(n,pos[0]),pos[1]); //0b11
		res(n) = m(k, 0b00) * vec(i0) + m(k, 0b01) * vec(i1) 
			+ m(k, 0b10)*vec(i2) + m(k, 0b11)*vec(i3);
	}
	return res;
}

template<typename T>
Eigen::VectorXcd apply_three_qubit(const Eigen::VectorXcd& vec,
		const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& m, std::array<uint32_t,3> pos)
{
	using detail::set;
	using detail::unset;
	Eigen::VectorXcd res(vec.size());
	for(uint32_t n = 0; n < vec.size(); ++n)
	{
		uint32_t k = (((n >> pos[2]) & 1u) << 2) |
			(((n >> pos[1]) & 1u) << 1) | ((n >> pos[0]) & 1u);
		uint32_t i0 = unset(unset(unset(n,pos[0]),pos[1]),pos[2]); //0b000
		uint32_t i1 = unset(unset(  set(n,pos[0]),pos[1]),pos[2]); //0b001
		uint32_t i2 = unset(  set(unset(n,pos[0]),pos[1]),pos[2]); //0b010
		uint32_t i3 = unset(  set(  set(n,pos[0]),pos[1]),pos[2]); //0b011
		uint32_t i4 =   set(unset(unset(n,pos[0]),pos[1]),pos[2]); //0b100
		uint32_t i5 =   set(unset(  set(n,pos[0]),pos[1]),pos[2]); //0b101
		uint32_t i6 =   set(  set(unset(n,pos[0]),pos[1]),pos[2]); //0b110
		uint32_t i7 =   set(  set(  set(n,pos[0]),pos[1]),pos[2]); //0b111

		res(n) = m(k, 0b000) * vec(i0) + m(k, 0b001) * vec(i1) 
			+ m(k, 0b010)*vec(i2) + m(k, 0b011)*vec(i3)
			+ m(k, 0b100)*vec(i4) + m(k, 0b101)*vec(i5)
			+ m(k, 0b110)*vec(i6) + m(k, 0b111)*vec(i7);
	}
	return res;
}
} //namespace qunn
