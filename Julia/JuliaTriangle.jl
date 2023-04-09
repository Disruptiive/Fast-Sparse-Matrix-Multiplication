using LinearAlgebra, SparseArrays, MAT, BenchmarkTools, MatrixMarket
using Base.Threads: @threads, @spawn


function tamudata(group, data)
	url = "https://suitesparse-collection-website.herokuapp.com/mat/$group/$data.mat"
	save_path = download(url)
	vars = matread(save_path)
    A = vars["Problem"]["A"]
end

function triangle_counting(A)
	return A .* (A * A)
end

#calculate value of M*M at (row,col)
function vector2vectorMult(row,col,M)
	row_start = M.colptr[row] #find start row index (symmetric matrix => CSR == CSC) 
	row_end = M.colptr[row+1] #find end row index
	col_start = M.colptr[col] #find start column index 
	col_end = M.colptr[col+1] #find start column index
	#tmp_i,tmp_j: pointers to check elements inside row and column 
	tmp_i = col_start
	tmp_j = row_start
	sum = 0 #sum = value of M*M at (row,col)
	#traverse through all elements inside the row and the column vector
	#if the elements are equal increment sum and pointers tmp_i,tmp_j by 1
	#else increment by 1 the pointer of the smaller element
	while (tmp_i<col_end && tmp_j<row_end) 
		if M.rowval[tmp_i] < M.rowval[tmp_j]
			tmp_i += 1
		elseif M.rowval[tmp_i] > M.rowval[tmp_j]
			tmp_j += 1
		else
			sum += 1
			tmp_i += 1
			tmp_j += 1
		end
	end
	return sum
end

#loop through all non zero elements and call vector2vectorMult() to find the value of M*M at each nz element
function sequential_triangle_counting(M)
	#loop through the columns of the CSC
	for i in 1:M.m
		col = i
		col_start = M.colptr[col]
		col_end = M.colptr[col+1]
		#loop through the rows of each column, call vector2vectorMult() for every non-zero element and update their value
		for j in col_start:col_end-1
			row = M.rowval[j]
			M.nzval[j] = vector2vectorMult(row,col,M)
		end
	end
	return M
end

#same as sequential_triangle_counting() but exterior for of nested loop if threaded
function threads_triangle_counting(M)
	@threads for i in 1:M.m
		col = i
		col_start = M.colptr[col]
		col_end = M.colptr[col+1]
		for j in col_start:col_end-1
			row = M.rowval[j]
			M.nzval[j] = vector2vectorMult(row,col,M)
		end
	end
	return M
end

#calculate triangle number by multiplying matrix M with a Mx1 vector of 1/2, adding all the elements and dividing the number by 3
function final_triangle_num(M)
	return (sum(M*ones(size(M,1))/2)/3)
end

let
	M = tamudata("DIMACS10","NACA0015")
	@time A=sequential_triangle_counting(M)
	println(final_triangle_num(A))	
end