import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import OptimizationService from '../services/optimization.service';
import '../styles/Optimizations.css';
import LoadingSkeleton from '../components/LoadingSkeleton';
import { useToast } from '../components/ToastProvider';

const Optimizations = () => {
  const [optimizations, setOptimizations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const { notify } = useToast();

  useEffect(() => {
    fetchOptimizations();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchOptimizations = async () => {
    try {
      setLoading(true);
      const response = await OptimizationService.getAll();
      setOptimizations(response);
      setError('');
      notify('Optimizations loaded', 'success', { autoClose: 1200 });
    } catch (err) {
      setError('Failed to load optimizations');
      notify('Failed to load optimizations', 'error');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (window.confirm('Are you sure you want to delete this optimization?')) {
      try {
        await OptimizationService.remove(id);
        setOptimizations(optimizations ? optimizations.filter(opt => opt._id !== id) : []);
        setError('');
        notify('Optimization deleted', 'success');
      } catch (err) {
        const msg = err?.response?.data?.msg || 'Failed to delete optimization';
        setError(msg);
        notify(msg, 'error');
        console.error('Delete optimization error:', err?.response?.data || err);
      }
    }
  };

  if (loading) {
    return (
      <div className="optimizations-container">
        <div className="optimizations-header">
          <h1>Optimizations</h1>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(350px,1fr))', gap: '1rem' }}>
          <LoadingSkeleton lines={4} />
          <LoadingSkeleton lines={4} />
          <LoadingSkeleton lines={4} />
        </div>
      </div>
    );
  }

  return (
    <div className="optimizations-container container mx-auto px-6 py-8">
      <div className="optimizations-header">
        <h1>Optimizations</h1>
        <Link to="/optimizations/new" className="btn btn-primary">
          <i className="fas fa-plus"></i> New Optimization
        </Link>
      </div>

      {error && <div className="alert alert-danger">{error}</div>}

      {!optimizations || optimizations.length === 0 ? (
        <div className="no-data">
          <p>No optimizations found. Create your first optimization!</p>
        </div>
      ) : (
        <div className="optimizations-grid">
          {optimizations && optimizations.length > 0 && optimizations.map(optimization => (
            <div key={optimization._id} className="optimization-card card card-hover rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 shadow hover:shadow-md transition">
              <div className="optimization-header">
                <div className="optimization-name">{optimization.name}</div>
                <div className="optimization-meta">
                  <span className="tag"><i className="fa fa-calendar"></i>{new Date(optimization.date).toLocaleDateString()}</span>
                  <span className="tag tag--primary"><i className="fa fa-route"></i>{optimization.routes ? optimization.routes.length : 0} routes</span>
                </div>
              </div>
              <div className="optimization-details">
                <p><i className="fas fa-road"></i> Total Distance: {Number(optimization?.totalDistance ?? 0).toFixed(2)} km</p>
              </div>
              <div className="optimization-actions flex gap-2">
                <Link to={`/optimizations/${optimization._id}`} className="btn btn-secondary rounded-lg px-4 py-2">
                  View Details
                </Link>
                <button
                  className="btn btn-danger rounded-lg px-4 py-2"
                  onClick={() => handleDelete(optimization._id)}
                >
                  <i className="fas fa-trash"></i> Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default Optimizations;