clear
close all
% clc

rng(4)

n = [ 0 2 2 2 1 1 ] ;
gam = 0.9 ;
a = 0.5 ;

N = sum(n) ;

q = randn(N,1) ;

%p = ones(N,1)/N ;
p = rand(N,1) ; p = p/sum(p) ;

r = [ 0 -10 0 -10 0 10 0 0 ]' ;

T = [ [ 0 0 0 0 1 0 0 0 ] ; ...
      [ 1 0 0 0 0 0 0 0 ] ; ...
      [ 0 0 1 0 0 0 0 0 ] ; ...
      [ 0 1 0 1 0 0 1 0 ] ; ...
      [ 0 0 0 0 0 1 0 1 ] ] ;

L = 6 ;
K = 1e3 ;


Q = zeros(N,K) ;

n_old = zeros(N,1) ;

for k = 1:K    

    P = zeros(size(T')) ;
    for s = 1:size(P,2)
        [~,i] = max( q( sum(n(1:s))+1 : sum(n(1:s))+n(s+1) ) ) ;        
        P(sum(n(1:s))+i,s) = 1 ;
    end
    A = P*T ;
    
    % initialize
    m_old = zeros(N,1) ;
    g_old = zeros(N,1) ;
    
    % simulate
    for l = 0:L
        g_new = g_old + diag( A^l * p ) * (eye(N)-gam*A')^-1 * (eye(N)-gam^(L+1-l)*A'^(L+1-l)) * r ;  
        m_new = m_old + A^l*p ;
        n_new = n_old + A^l*p ;
        
        g_old = g_new ;
        m_old = m_new ;
        n_old = n_new ;
    end
    
    q = q + diag(n_old)^-1*( g_old - diag(m_old)*q) ;
    %q = q + (diag(m_old)^-1*g_old - q) ;
    %q = q + a^k*( g_old - q) ;
    
    Q(:,k) = q ;
end

plot(1:K, Q)

q

%%

Q = zeros(N,K) ;

for k = 1:K    

    P = zeros(size(T')) ;
    for s = 1:size(P,2)
        
        % find index of maximal value and break random ties
    %     M = -inf ;
    %     for a = 1:n(s+1)
    %         if q() > M
    %             
    %         end
    %     end
        [~,i] = max( q( sum(n(1:s))+1 : sum(n(1:s))+n(s+1) ) ) ;
        
        P(sum(n(1:s))+i,s) = 1 ;
    end
    
    % initialize
    z_old = zeros(N,1) ;
    z_old(randsample(N, 1, true, p)) = 1 ;
    %z_old = ones(N,1) ;
    e_old = zeros(N,1) ;
    g_old = zeros(N,1) ;
    m_old = zeros(N,1) ;
    
    % simulate
    for l = 1:L
        z_new = P*T*z_old ;
        e_new = gam*e_old + z_old ;
        g_new = g_old + e_new*z_old'*r ;  
        %m_new = min(m_old + z_old , 1) ;
        m_new = m_old + z_old ;
        
        z_old = z_new ;
        e_old = e_new ;
        g_old = g_new ;
        m_old = m_new ;
    end
    
    a = 0.99*a ;
    q = q + a*diag(m_old)*(g_old-q) ;
    
    Q(:,k) = q ;
end

plot(1:K, Q)


%%

p = randn(N,1) ;
r = randn(N,1) ;
A = randn(N,N) ;

diag( (A*p) .* (A'*r) )
diag(p) * A' * A' * diag(r)


