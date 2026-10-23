// API configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Safely format FastAPI error details (string, dict, or list of error objects)
 */
export function formatApiErrorDetail(detail) {
  if (detail == null) return null;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map(d => (typeof d === 'object' && d.msg) ? d.msg : JSON.stringify(d)).join('; ');
  }
  if (typeof detail === 'object') {
    const msg = detail.message || detail.detail || 'Request failed';
    const errs = detail.errors;
    if (Array.isArray(errs) && errs.length) return `${msg}: ${errs.join('; ')}`;
    return msg;
  }
  return String(detail);
}

/**
 * Get the stored access token
 */
export const getToken = () => {
  return localStorage.getItem('access_token');
};

/**
 * Set the access token
 */
export const setToken = (token) => {
  localStorage.setItem('access_token', token);
};

/**
 * Remove the access token
 */
export const removeToken = () => {
  localStorage.removeItem('access_token');
};

/**
 * Make an API request with authentication
 */
export const apiRequest = async (endpoint, options = {}) => {
  const token = getToken();
  const url = `${API_BASE_URL}${endpoint}`;
  
  const headers = {
    ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, config);
    
    // Check if response has JSON content
    const contentType = response.headers.get('content-type');
    let data;
    
    if (contentType && contentType.includes('application/json')) {
      data = await response.json();
    } else {
      const text = await response.text();
      data = { detail: text || `HTTP error! status: ${response.status}` };
    }

    if (!response.ok) {
      const errMsg = formatApiErrorDetail(data.detail) || `HTTP error! status: ${response.status}`;
      throw new Error(errMsg);
    }

    return { success: true, data };
  } catch (error) {
    // Handle network errors or JSON parsing errors
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      return { 
        success: false, 
        error: 'Network error. Please check if the backend server is running.' 
      };
    }
    return { 
      success: false, 
      error: error.message || 'An error occurred while making the request' 
    };
  }
};

/**
 * Auth API functions
 */
export const authAPI = {
  // Login
  login: async (email, password) => {
    return apiRequest('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  },

  // Patient signup
  signupPatient: async (userData) => {
    return apiRequest('/auth/signup/patient', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  // Doctor signup
  signupDoctor: async (userData) => {
    return apiRequest('/auth/signup/doctor', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  },

  // Get current user
  getCurrentUser: async () => {
    return apiRequest('/auth/me', {
      method: 'GET',
    });
  },
};
