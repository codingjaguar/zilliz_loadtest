package mocks

import (
	"context"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// MockClient is a mock implementation of client.Client for testing
// It implements only the methods we actually use in our code
type MockClient struct {
	SearchFunc              func(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error)
	InsertFunc              func(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) (entity.Column, error)
	DescribeCollectionFunc  func(ctx context.Context, collectionName string) (*entity.Collection, error)
	GetCollectionStatisticsFunc func(ctx context.Context, collectionName string) (map[string]string, error)
	DescribeIndexFunc      func(ctx context.Context, collectionName string, fieldName string) ([]entity.Index, error)
	HasCollectionFunc      func(ctx context.Context, collectionName string) (bool, error)
	CreateCollectionFunc    func(ctx context.Context, schema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error
	CreateIndexFunc         func(ctx context.Context, collectionName string, fieldName string, idx entity.Index, async bool, opts ...client.IndexOption) error
	GetIndexStateFunc       func(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) (entity.IndexState, error)
	ListCollectionsFunc     func(ctx context.Context) ([]*entity.Collection, error)
	CloseFunc               func() error
}

func (m *MockClient) Search(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	if m.SearchFunc != nil {
		return m.SearchFunc(ctx, collectionName, partitionNames, expr, outputFields, vectors, vectorField, metricType, topK, sp, opts...)
	}
	return []client.SearchResult{}, nil
}

func (m *MockClient) Insert(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) (entity.Column, error) {
	if m.InsertFunc != nil {
		return m.InsertFunc(ctx, collectionName, partitionName, columns...)
	}
	return nil, nil
}

func (m *MockClient) DescribeCollection(ctx context.Context, collectionName string) (*entity.Collection, error) {
	if m.DescribeCollectionFunc != nil {
		return m.DescribeCollectionFunc(ctx, collectionName)
	}
	return &entity.Collection{}, nil
}

func (m *MockClient) GetCollectionStatistics(ctx context.Context, collectionName string) (map[string]string, error) {
	if m.GetCollectionStatisticsFunc != nil {
		return m.GetCollectionStatisticsFunc(ctx, collectionName)
	}
	return map[string]string{"row_count": "1000"}, nil
}

func (m *MockClient) DescribeIndex(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) ([]entity.Index, error) {
	if m.DescribeIndexFunc != nil {
		return m.DescribeIndexFunc(ctx, collectionName, fieldName)
	}
	return []entity.Index{}, nil
}
func (m *MockClient) GetIndexState(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) (entity.IndexState, error) {
	if m.GetIndexStateFunc != nil {
		return m.GetIndexStateFunc(ctx, collectionName, fieldName, opts...)
	}
	return 0, nil
}

func (m *MockClient) Close() error {
	if m.CloseFunc != nil {
		return m.CloseFunc()
	}
	return nil
}

// Stub out all other client.Client interface methods with no-ops
// These are required by the interface but not used in our code

func (m *MockClient) AddUserRole(ctx context.Context, username string, roleName string) error { return nil }
func (m *MockClient) AlterAlias(ctx context.Context, alias string, collectionName string) error { return nil }
func (m *MockClient) AlterCollection(ctx context.Context, collectionName string, opts ...entity.CollectionAttribute) error { return nil }
func (m *MockClient) AlterIndex(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) error { return nil }
func (m *MockClient) AlterUser(ctx context.Context, username string, opts ...interface{}) error { return nil }
func (m *MockClient) CheckHealth(ctx context.Context) error { return nil }
func (m *MockClient) CreateAlias(ctx context.Context, alias string, collectionName string) error { return nil }
func (m *MockClient) CreateCollection(ctx context.Context, schema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error {
	if m.CreateCollectionFunc != nil {
		return m.CreateCollectionFunc(ctx, schema, shardsNum, opts...)
	}
	return nil
}
func (m *MockClient) NewCollection(ctx context.Context, collName string, dimension int64, opts ...client.CreateCollectionOption) error { return nil }
func (m *MockClient) CreateCredential(ctx context.Context, username string, password string) error { return nil }
func (m *MockClient) CreateDatabase(ctx context.Context, dbName string) error { return nil }
func (m *MockClient) DropDatabase(ctx context.Context, dbName string) error { return nil }
func (m *MockClient) CreateIndex(ctx context.Context, collectionName string, fieldName string, idx entity.Index, async bool, opts ...client.IndexOption) error {
	if m.CreateIndexFunc != nil {
		return m.CreateIndexFunc(ctx, collectionName, fieldName, idx, async, opts...)
	}
	return nil
}
func (m *MockClient) CreatePartition(ctx context.Context, collectionName string, partitionName string) error { return nil }
func (m *MockClient) CreateRole(ctx context.Context, roleName string) error { return nil }
func (m *MockClient) CreateUser(ctx context.Context, username string, password string) error { return nil }
func (m *MockClient) DeleteByPks(ctx context.Context, collectionName string, partitionName string, ids entity.Column) error { return nil }
func (m *MockClient) DeleteCredential(ctx context.Context, username string) error { return nil }
func (m *MockClient) DeleteDatabase(ctx context.Context, dbName string) error { return nil }
func (m *MockClient) DeletePartition(ctx context.Context, collectionName string, partitionName string) error { return nil }
func (m *MockClient) DeleteRole(ctx context.Context, roleName string) error { return nil }
func (m *MockClient) DropRole(ctx context.Context, name string) error { return nil }
func (m *MockClient) DeleteUser(ctx context.Context, username string) error { return nil }
func (m *MockClient) DropAlias(ctx context.Context, alias string) error { return nil }
func (m *MockClient) DropCollection(ctx context.Context, collectionName string) error { return nil }
func (m *MockClient) DropIndex(ctx context.Context, collectionName string, fieldName string, opts ...client.IndexOption) error { return nil }
func (m *MockClient) DropPartition(ctx context.Context, collectionName string, partitionName string) error { return nil }
func (m *MockClient) Flush(ctx context.Context, collectionName string, async bool) error { return nil }
func (m *MockClient) GetLoadingProgress(ctx context.Context, collectionName string, partitionNames []string) (int64, error) { return 0, nil }
func (m *MockClient) GetPersistentSegmentInfo(ctx context.Context, collectionName string) ([]*entity.Segment, error) { return nil, nil }
func (m *MockClient) GetQuerySegmentInfo(ctx context.Context, collectionName string) ([]*entity.Segment, error) { return nil, nil }
func (m *MockClient) GetReplicas(ctx context.Context, collectionName string) ([]*entity.ReplicaGroup, error) { return nil, nil }
func (m *MockClient) HasCollection(ctx context.Context, collectionName string) (bool, error) {
	if m.HasCollectionFunc != nil {
		return m.HasCollectionFunc(ctx, collectionName)
	}
	return false, nil
}
func (m *MockClient) HasPartition(ctx context.Context, collectionName string, partitionName string) (bool, error) { return false, nil }
func (m *MockClient) ListAliases(ctx context.Context, collectionName string) ([]string, error) { return nil, nil }
func (m *MockClient) ListCollections(ctx context.Context) ([]*entity.Collection, error) {
	if m.ListCollectionsFunc != nil {
		return m.ListCollectionsFunc(ctx)
	}
	return nil, nil
}
func (m *MockClient) ListCredentialUsers(ctx context.Context) ([]string, error) { return nil, nil }
func (m *MockClient) ListCredUsers(ctx context.Context) ([]string, error) { return nil, nil }
func (m *MockClient) ListDatabases(ctx context.Context) ([]entity.Database, error) { return nil, nil }
func (m *MockClient) ListIndexes(ctx context.Context, collectionName string) ([]entity.Index, error) { return nil, nil }
func (m *MockClient) ShowPartitions(ctx context.Context, collectionName string) ([]*entity.Partition, error) { return nil, nil }
func (m *MockClient) ListRoles(ctx context.Context) ([]entity.Role, error) { return nil, nil }
func (m *MockClient) LoadCollection(ctx context.Context, collectionName string, async bool, opts ...client.LoadCollectionOption) error { return nil }
func (m *MockClient) LoadPartitions(ctx context.Context, collectionName string, partitionNames []string, async bool) error { return nil }
func (m *MockClient) QueryByPks(ctx context.Context, collectionName string, partitionNames []string, ids entity.Column, fieldNames []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) { return nil, nil }
func (m *MockClient) ReleaseCollection(ctx context.Context, collectionName string) error { return nil }
func (m *MockClient) ReleasePartitions(ctx context.Context, collectionName string, partitionNames []string) error { return nil }
func (m *MockClient) RemoveUserRole(ctx context.Context, username string, roleName string) error { return nil }
func (m *MockClient) RenameCollection(ctx context.Context, oldName string, newName string) error { return nil }
func (m *MockClient) UpdateCredential(ctx context.Context, username string, oldPassword string, newPassword string) error { return nil }
func (m *MockClient) Upsert(ctx context.Context, collectionName string, partitionName string, columns ...entity.Column) (entity.Column, error) { return nil, nil }
func (m *MockClient) UsingDatabase(ctx context.Context, dbName string) error { return nil }
func (m *MockClient) BulkInsert(ctx context.Context, collName string, partitionName string, files []string, opts ...client.BulkInsertOption) (int64, error) { return 0, nil }
func (m *MockClient) GetBulkInsertState(ctx context.Context, taskID int64) (*entity.BulkInsertTaskState, error) { return nil, nil }
func (m *MockClient) ListBulkInsertTasks(ctx context.Context, collName string, limit int64) ([]*entity.BulkInsertTaskState, error) { return nil, nil }
func (m *MockClient) Query(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) { return nil, nil }
func (m *MockClient) Get(ctx context.Context, collectionName string, ids entity.Column, opts ...client.GetOption) (client.ResultSet, error) { return nil, nil }
func (m *MockClient) CalcDistance(ctx context.Context, collName string, partitions []string, metricType entity.MetricType, leftVectors entity.Column, rightVectors entity.Column) (entity.Column, error) { return nil, nil }
func (m *MockClient) CreateCollectionByRow(ctx context.Context, row entity.Row, shardNum int32) error { return nil }
func (m *MockClient) InsertByRows(ctx context.Context, collName string, partitionName string, rows []entity.Row) (entity.Column, error) { return nil, nil }
func (m *MockClient) InsertRows(ctx context.Context, collName string, partitionName string, rows []interface{}) (entity.Column, error) { return nil, nil }
func (m *MockClient) ManualCompaction(ctx context.Context, collName string, toleranceDuration time.Duration) (int64, error) { return 0, nil }
func (m *MockClient) GetCompactionState(ctx context.Context, id int64) (entity.CompactionState, error) { return 0, nil }
func (m *MockClient) GetCompactionStateWithPlans(ctx context.Context, id int64) (entity.CompactionState, []entity.CompactionPlan, error) { return 0, nil, nil }
func (m *MockClient) Grant(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error { return nil }
func (m *MockClient) Revoke(ctx context.Context, role string, objectType entity.PriviledgeObjectType, object string) error { return nil }
func (m *MockClient) GetLoadState(ctx context.Context, collectionName string, partitionNames []string) (entity.LoadState, error) { return entity.LoadStateNotExist, nil }
func (m *MockClient) ListResourceGroups(ctx context.Context) ([]string, error) { return nil, nil }
func (m *MockClient) CreateResourceGroup(ctx context.Context, rgName string) error { return nil }
func (m *MockClient) DescribeResourceGroup(ctx context.Context, rgName string) (*entity.ResourceGroup, error) { return nil, nil }
func (m *MockClient) DropResourceGroup(ctx context.Context, rgName string) error { return nil }
func (m *MockClient) TransferNode(ctx context.Context, sourceRg, targetRg string, nodesNum int32) error { return nil }
func (m *MockClient) TransferReplica(ctx context.Context, sourceRg, targetRg string, collectionName string, replicaNum int64) error { return nil }
func (m *MockClient) GetVersion(ctx context.Context) (string, error) { return "", nil }
func (m *MockClient) ListUsers(ctx context.Context) ([]entity.User, error) { return nil, nil }
